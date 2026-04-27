import os
import sys
import time
import re
import json

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Phase_1.loaders.config_loader  import get_settings, get_prompts
from Phase_1.core.vlm_engine        import VLMEngine
from Phase_1.loaders.registry       import RegistryManager, STATUS_PENDING, STATUS_P1_OK, STATUS_P2_OK, STATUS_ERROR
from Phase_1.utils.json_handler     import save_json_with_ids, generate_image_id, canonicalize_fields

# ─────────────────────────────────────────────────────────────────
#  Helpers & Debug Logs
# ─────────────────────────────────────────────────────────────────
def get_debug_dir():
    d = os.path.join(project_root, "Phase_1", "debug_outputs")
    os.makedirs(d, exist_ok=True)
    return d

def build_debug_filename(prefix, disease_name, image_name):
    safe = lambda s: re.sub(r'[^\w\-_.]', '_', s or "")
    ts   = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{safe(disease_name)}__{safe(image_name)}__{ts}.txt"

def save_debug_file(prefix, disease_name, image_name, sys_prompt, user_prompt, raw_response, model_name="GPT-5"):
    debug_dir = get_debug_dir()
    path = os.path.join(debug_dir, build_debug_filename(prefix, disease_name, image_name))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n"
                f"[{prefix}] {model_name}\n"
                f"{'='*60}\n"
                f"BỆNH  : {disease_name}\n"
                f"ẢNH   : {image_name}\n"
                f"TIME  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*60}\n"
                f"\n## SYSTEM PROMPT ##\n{sys_prompt}\n"
                f"\n## USER PROMPT ##\n{user_prompt}\n"
                f"\n{'='*60}\n"
                f"## RAW RESPONSE ##\n{'='*60}\n"
                f"{raw_response}\n")

# ─────────────────────────────────────────────────────────────────
#  Prompt builders
# ─────────────────────────────────────────────────────────────────
def build_p1_prompt(p1_config):
    user_template = p1_config.get('user_template', '')
    examples      = p1_config.get('few_shot_examples', [])

    if examples:
        user_template += "\n\n--- VÍ DỤ MINH HỌA ---\n"
        for ex_item in examples:
            for _, ex_val in ex_item.items():
                img_name = os.path.basename(ex_val.get('image_name', ''))
                user_template += (f"[{img_name}]\n{ex_val.get('expected_output', '').strip()}\n\n")
        user_template += "--> BÂY GIỜ LÀ LƯỢT CỦA BẠN:\n"
    return user_template

def build_p2_prompt(p2_config, phase1_output, disease_name, knowledge):
    template = p2_config.get('user_template', '')
    
    # [SỬA LỖI .format()]: Dùng replace để tránh crash nếu dữ liệu có ngoặc nhọn {}
    formatted = template.replace("{phase1_qa_output}", phase1_output) \
                        .replace("{disease_name}", disease_name) \
                        .replace("{disease_knowledge}", knowledge or "(Không có kiến thức)")

    examples = p2_config.get('few_shot_examples', [])
    if examples:
        formatted += "\n\n--- VÍ DỤ MINH HỌA ---\n"
        for ex_item in examples:
            for _, ex_val in ex_item.items():
                formatted += (
                    f"[{ex_val.get('disease_name', '')}]\n"
                    f"QA Đầu vào:\n{ex_val.get('phase1_qa_output', '').strip()}\n\n"
                    f"JSON Đầu ra:\n{ex_val.get('expected_json', '').strip()}\n\n"
                )
        formatted += f"--> TRÍCH XUẤT JSON CHO BỆNH: {disease_name}\n"
    return formatted

# ─────────────────────────────────────────────────────────────────
#  Retry wrapper & Knowledge Finder
# ─────────────────────────────────────────────────────────────────
def _call_with_retry(call_fn, label, max_retries=3, retry_delay=5):
    for attempt in range(1, max_retries + 1):
        print(f"\n  [{label}] Lần #{attempt}/{max_retries}...")
        result = call_fn()
        if result and result.strip() and "lỗi" not in result.lower() and "error" not in result.lower():
            if attempt > 1:
                print(f"  [{label}] ✅ Thành công ở lần #{attempt}")
            return result
        print(f"  [{label}] ⚠️ Lần #{attempt} thất bại. Trả về: {result[:50]}")
        if attempt < max_retries:
            print(f"  [{label}] ⏳ Đợi {retry_delay}s trước retry...")
            time.sleep(retry_delay)
    print(f"  [{label}] ❌ Tất cả retry đều thất bại.")
    return result if result else "LỖI: Hết retry"

def find_knowledge_file(contents_dir: str, disease_folder: str):
    if not os.path.isdir(contents_dir):
        return None
    target = disease_folder.lower().strip()
    for fname in sorted(os.listdir(contents_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        part = re.sub(r"^.*?\-\s*", "", fname)
        part = re.sub(r"\.txt$", "", part, flags=re.IGNORECASE).strip()
        if part.lower() == target or target in part.lower() or part.lower() in target:
            return os.path.join(contents_dir, fname)
    return None

def extract_disease_name(knowledge_filename: str) -> str:
    name = re.sub(r"^.*?\-\s*", "", knowledge_filename)
    name = re.sub(r"\.(txt|TXT)$", "", name).strip()
    return name

# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SETTINGS = get_settings()
    PROMPTS_CFG = get_prompts()

    DATASET_DIR = os.path.join(project_root, SETTINGS["paths"]["data_raw"])
    CONTENTS_DIR = os.path.join(DATASET_DIR, "contents")
    IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    OUTPUT_DIR = os.path.join(project_root, SETTINGS["paths"]["data_consensus"])
    REGISTRY_PATH = os.path.join(project_root, "Phase_1", "master_registry.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    registry = RegistryManager(REGISTRY_PATH)
    registry.discover_dataset(IMAGES_DIR, CONTENTS_DIR)

    vlm_cfg = SETTINGS["models"]["vlm_main"]
    engine = VLMEngine()
    engine.load_model(provider=vlm_cfg["provider"], model_id=vlm_cfg["model_id"])

    # LẤY DANH SÁCH CHƯA HOÀN THÀNH (Bao gồm ERROR) VÀ SẮP XẾP Z -> A
    pending_list = registry.get_pending()
    pending_list.sort(key=lambda x: (x["disease_folder"], x["image_path"]), reverse=True)

    print(f"\n🚀 TÌM THẤY {len(pending_list)} ẢNH CẦN CHẠY. BẮT ĐẦU CHẠY NGƯỢC TỪ Z -> A...")

    for entry in pending_list:
        img_rel = entry["image_path"]
        folder = entry["disease_folder"]
        img_abs = os.path.join(IMAGES_DIR, img_rel)
        
        knowledge_path = find_knowledge_file(CONTENTS_DIR, folder)
        
        if not knowledge_path or not os.path.exists(knowledge_path):
            print(f"⚠️ BỎ QUA '{folder}': Không tìm thấy file kiến thức bệnh cho '{folder}'")
            registry.update_status(img_rel, STATUS_ERROR, {"error_log": "Missing txt file"})
            continue

        disease_name = extract_disease_name(os.path.basename(knowledge_path))
        
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            disease_knowledge = f.read().strip()

        print(f"\n{'='*55}\n🚀 Đang xử lý: {img_rel} (Bệnh: {disease_name})")

        # --- Phase 1: Quan sát ---
        sys_p1 = PROMPTS_CFG["phase1_observation_qa"]["system_instruction"]
        usr_p1 = build_p1_prompt(PROMPTS_CFG["phase1_observation_qa"])
        
        phase1_raw = _call_with_retry(
            lambda: engine.call_vlm(sys_p1, usr_p1, image_path=img_abs),
            label="Phase 1",
            max_retries=3
        )
        
        if phase1_raw.startswith("LỖI"):
            print(f"❌ Lỗi Phase 1: {phase1_raw}")
            registry.update_phase(img_rel, "phase1", STATUS_ERROR, raw_text=phase1_raw, error="Lỗi gọi API P1")
            continue
            
        registry.update_phase(img_rel, "phase1", STATUS_P1_OK, raw_text=phase1_raw)
        
        if SETTINGS["pipeline"].get("debug_save_raw"):
            save_debug_file("P1", disease_name, os.path.basename(img_abs), sys_p1, usr_p1, phase1_raw, vlm_cfg["model_id"])

        # --- Phase 2: Chuẩn hóa JSON (KHÔNG gửi ảnh) ---
        sys_p2 = PROMPTS_CFG["phase2_json_standardization"]["system_instruction"]
        usr_p2 = build_p2_prompt(PROMPTS_CFG["phase2_json_standardization"], phase1_raw, disease_name, disease_knowledge)
        
        phase2_raw = _call_with_retry(
            lambda: engine.call_vlm(sys_p2, usr_p2, image_path=None),
            label="Phase 2",
            max_retries=3
        )
        
        parsed = engine.extract_json(phase2_raw)
        
        if not isinstance(parsed, dict) or "error" in parsed:
            print(f"❌ Lỗi Phase 2 (Parse JSON): {parsed.get('error', 'Unknown Error')}")
            registry.update_phase(img_rel, "phase2", STATUS_ERROR, raw_text=phase2_raw, error="Parse JSON thất bại")
            continue

        if SETTINGS["pipeline"].get("debug_save_raw"):
            save_debug_file("P2", disease_name, os.path.basename(img_abs), sys_p2, usr_p2, phase2_raw, vlm_cfg["model_id"])

        # --- Lưu File Final ---
        je_data = canonicalize_fields(parsed.get("JSON_EXTRACTION", parsed))
        image_id = generate_image_id(os.path.basename(img_abs), folder)
        
        save_path = os.path.join(OUTPUT_DIR, folder, f"{os.path.splitext(os.path.basename(img_abs))[0]}_final.json")
        save_json_with_ids(
            {"JSON_EXTRACTION": je_data}, 
            save_path, 
            image_id=image_id, 
            source=vlm_cfg["model_id"]
        )
        
        # Đánh dấu hoàn thành toàn bộ
        registry.update_phase(img_rel, "phase2", STATUS_P2_OK, raw_text=phase2_raw)
        print(f"✅ Hoàn thành: {save_path}")

        time.sleep(5) # Delay tránh rate limit