"""
run_test_demo.py — Chạy thử nghiệm luồng GPT Đơn (Phase 1 + Phase 2)
Dùng để kiểm tra nhanh API và Prompts trên vài case trong thư mục test_demo.
"""

import os
import sys
import json
import time
import re

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Phase_1.loaders.config_loader  import get_settings, get_prompts
from Phase_1.core.vlm_engine        import VLMEngine
from Phase_1.utils.json_handler     import save_json_with_ids, generate_image_id, canonicalize_fields

# ─────────────────────────────────────────────────────────────────
#  Cấu hình Test
# ─────────────────────────────────────────────────────────────────
WAIT_BETWEEN_PHASES = 3
WAIT_AFTER_CASE     = 5
RETRY_COUNT         = 3
RETRY_DELAY         = 5

# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def get_debug_dir():
    d = os.path.join(project_root, "Phase_1", "debug_outputs")
    os.makedirs(d, exist_ok=True)
    return d

def build_debug_path(prefix, disease_name, image_name):
    safe = lambda s: re.sub(r'[^\w\-_.]', '_', s or "")
    ts   = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(get_debug_dir(), f"{prefix}_{safe(disease_name)}__{safe(image_name)}__{ts}.txt")

def save_debug(prefix, disease_name, image_name, sys_prompt, user_prompt, raw_text, model_id):
    path = build_debug_path(prefix, disease_name, image_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n"
                f"[{prefix}] MODEL: {model_id}\n"
                f"{'='*60}\n"
                f"BỆNH  : {disease_name}\n"
                f"ẢNH   : {image_name}\n"
                f"TIME  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*60}\n\n"
                f"## SYSTEM ##\n{sys_prompt}\n\n"
                f"## USER ##\n{user_prompt}\n\n"
                f"{'='*60}\n## RAW ##\n{'='*60}\n{raw_text}\n")
    print(f"[DEBUG] Đã lưu response thô → {path}")

def call_with_retry(call_fn, label, max_retries=RETRY_COUNT, retry_delay=RETRY_DELAY):
    for attempt in range(1, max_retries + 1):
        print(f"\n  [{label}] Lần thử #{attempt}/{max_retries}...")
        result = call_fn()
        # Kiểm tra lỗi "blocked" từ Proxy API
        if result and not result.startswith("LỖI") and "blocked" not in result.lower():
            if attempt > 1:
                print(f"  [{label}] ✅ Thành công ở lần #{attempt}")
            return result
        print(f"  [{label}] ⚠️ Thất bại: {str(result)[:100]}")
        if attempt < max_retries:
            print(f"  [{label}] ⏳ Đợi {retry_delay}s trước retry...")
            time.sleep(retry_delay)
    return result if result else "LỖI: Hết retry"

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
    # Sử dụng replace thay vì .format() để tránh lỗi với ký tự ngoặc nhọn {} trong dữ liệu bệnh
    template = p2_config.get('user_template', '')
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
#  Main Test Logic
# ─────────────────────────────────────────────────────────────────
def run_demo(image_path, disease_txt_path, prompts_config, output_dir, engine, vlm_cfg):
    txt_filename = os.path.basename(disease_txt_path)
    # Rút trích tên bệnh linh hoạt
    disease_name = re.sub(r'^.*?(\-)\s*', '', txt_filename)
    disease_name = re.sub(r'\.txt$', '', disease_name, flags=re.IGNORECASE).strip()
    img_filename = os.path.basename(image_path)

    print(f"\n{'='*60}")
    print(f"🚀 TEST DEMO")
    print(f"   Bệnh : {disease_name}")
    print(f"   Ảnh  : {img_filename}")
    print(f"   Model: {vlm_cfg['model_id']}")
    print(f"{'='*60}")

    with open(disease_txt_path, 'r', encoding='utf-8') as f:
        disease_knowledge = f.read().strip()

    try:
        # ── Phase 1: Quan sát (Gửi ảnh) ─────────────────────────────────────────
        sys_p1 = prompts_config["phase1_observation_qa"]["system_instruction"]
        usr_p1 = build_p1_prompt(prompts_config["phase1_observation_qa"])

        print(f"\n[Phase 1] 🔍 Đang quan sát ảnh bằng {vlm_cfg['model_id']}...")
        phase1_raw = call_with_retry(
            lambda: engine.call_vlm(sys_p1, usr_p1, image_path=image_path),
            "Phase1"
        )

        save_debug("TEST_P1", disease_name, img_filename, sys_p1, usr_p1, phase1_raw, vlm_cfg['model_id'])

        if phase1_raw.startswith("LỖI") or "blocked" in phase1_raw.lower():
            print(f"❌ [Phase 1] Bị chặn hoặc lỗi: {phase1_raw}")
            return

        print(f"\n{'─'*50}\n[Phase 1] ✅ KẾT QUẢ:\n{phase1_raw}\n{'─'*50}")

        print(f"\n⏳ Đợi {WAIT_BETWEEN_PHASES}s trước Phase 2...")
        time.sleep(WAIT_BETWEEN_PHASES)

        # ── Phase 2: Sinh JSON (KHÔNG gửi ảnh - để giảm nguy cơ bị chặn nội dung) ────────
        sys_p2 = prompts_config["phase2_json_standardization"]["system_instruction"]
        usr_p2 = build_p2_prompt(
            prompts_config["phase2_json_standardization"],
            phase1_raw, disease_name, disease_knowledge,
        )

        print(f"[Phase 2] 📋 Gọi {vlm_cfg['model_id']} — chuẩn hóa JSON (Text Only)...")
        phase2_raw = call_with_retry(
            lambda: engine.call_vlm(sys_p2, usr_p2, image_path=None), # Không gửi ảnh ở đây
            "Phase2"
        )

        save_debug("TEST_P2", disease_name, img_filename, sys_p2, usr_p2, phase2_raw, vlm_cfg['model_id'])

        parsed    = engine.extract_json(phase2_raw)
        if "error" in parsed:
            print(f"⚠️ Lỗi Parse JSON: {parsed.get('error')}")
            return

        je_data = canonicalize_fields(parsed.get("JSON_EXTRACTION", parsed))
        
        print(f"\n{'─'*50}\n[Phase 2] ✅ JSON HOÀN CHỈNH:\n{json.dumps(je_data, indent=2, ensure_ascii=False)}\n{'─'*50}")

        # ── Lưu file cuối cùng ─────────────────────────────────────────
        img_id = generate_image_id(img_filename)
        save_path = os.path.join(output_dir, f"{disease_name}_{img_filename}_final.json")
        
        save_json_with_ids(
            {"JSON_EXTRACTION": je_data}, 
            save_path, 
            image_id=img_id, 
            source=vlm_cfg['model_id']
        )
        print(f"✅ Đã lưu JSON chuẩn → {save_path}")

    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")

# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start = time.time()
    try:
        SETTINGS    = get_settings()
        PROMPTS_CFG = get_prompts()
    except Exception as e:
        print(f"❌ Lỗi load config: {e}")
        sys.exit(1)

    TEST_DEMO_DIR = os.path.join(project_root, "Phase_1", "test_demo")
    OUTPUT_DIR    = os.path.join(project_root, "Phase_1", "output", "test_demo")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    engine = VLMEngine()
    vlm_cfg = SETTINGS["models"]["vlm_main"]
    engine.load_model(provider=vlm_cfg["provider"], model_id=vlm_cfg["model_id"])

    # Quét ảnh và text
    test_cases = []
    for root, _, files in os.walk(TEST_DEMO_DIR):
        for fname in sorted(files):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, fname)
                txt_name = next((f for f in files if f.lower().endswith('.txt')), None)
                if txt_name:
                    test_cases.append((img_path, os.path.join(root, txt_name)))

    for i, (img_path, txt_path) in enumerate(test_cases, 1):
        run_demo(img_path, txt_path, PROMPTS_CFG, OUTPUT_DIR, engine, vlm_cfg)
        if i < len(test_cases):
            time.sleep(WAIT_AFTER_CASE)

    print(f"\n✅ HOÀN TẤT — {(time.time() - start) / 60:.1f} phút")