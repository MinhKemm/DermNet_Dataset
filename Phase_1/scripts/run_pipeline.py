"""
run_pipeline.py — Phase 1 Full Pipeline trên dermnet-output

Luồng xử lý:
  Phase 1a: VLM1 (Claude Opus 4.6) → JSON mô tả hình thái
  Phase 1b: VLM2 (GPT-4o)          → JSON mô tả hình thái  [bạn của bạn chạy]
  Judge   : Gemma 4                  → Hợp nhất & sinh JSON cuối cùng

Thực tế hiện tại: chỉ chạy VLM1 (Claude Opus 4.6).
Cấu trúc VLM2 + Judge giữ nguyên để push GitHub đầy đủ.

Dataset:
  dermnet-output/
  ├── images/<DiseaseName>/*.jpg, *.png
  └── contents/<Toàn bộ nội dung - DiseaseName.txt>

Chạy:
    cd /Users/binhminh/Desktop/DermNet_Dataset
    RUN_MODE=VLM1_ONLY python Phase_1/scripts/run_pipeline.py
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

from Phase_1.loaders.config_loader import get_settings, get_prompts
from Phase_1.core.vlm_engine   import VLMEngine
from Phase_1.core.judge_engine import JudgeEngine
from Phase_1.utils.json_handler import save_json_ordered


# ─────────────────────────────────────────────────────────────────
#  Chế độ chạy
# ─────────────────────────────────────────────────────────────────
# "BOTH"      : VLM1 + VLM2 → Judge (khi có đủ JSON)
# "VLM1_ONLY" : Chỉ Claude Opus → lưu trực tiếp  ← hiện tại
# "VLM2_ONLY" : Chỉ GPT-4o    → lưu trực tiếp
RUN_MODE = os.environ.get("RUN_MODE", "VLM1_ONLY").upper()


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def get_debug_dir():
    d = os.path.join(project_root, "Phase_1", "debug_outputs")
    os.makedirs(d, exist_ok=True)
    return d


def build_debug_filename(prefix, disease_name, image_name):
    safe = lambda s: re.sub(r'[^\w\-_.]', '_', s or "")
    ts   = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{safe(disease_name)}__{safe(image_name)}__{ts}.txt"


def save_debug_file(prefix, disease_name, image_name,
                   sys_prompt, user_prompt, raw_response):
    debug_dir = get_debug_dir()
    path = os.path.join(debug_dir, build_debug_filename(prefix, disease_name, image_name))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n"
                f"[{prefix}] Claude Opus 4.6 – Phase 1/2 Response\n"
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
    print(f"[DEBUG] Đã lưu → {path}")


# ─────────────────────────────────────────────────────────────────
#  Prompt builders (few-shot từ YAML)
# ─────────────────────────────────────────────────────────────────
def build_p1_prompt(p1_config):
    user_template = p1_config.get('user_template', '')
    examples      = p1_config.get('few_shot_examples', [])

    if examples:
        user_template += "\n\n--- CÁC VÍ DỤ MINH HỌA ---\n"
        for ex_item in examples:
            for _, ex_val in ex_item.items():
                img_name = os.path.basename(ex_val.get('image_name', ''))
                user_template += (
                    f"Ví dụ ({img_name}):\n"
                    f"{ex_val.get('expected_output', '').strip()}\n\n"
                )
        user_template += ("--> BÂY GIỜ LÀ LƯỢT CỦA BẠN "
                          "(Vui lòng chỉ trả lời 5 mục):")
    return user_template


def build_p2_prompt(p2_config, phase1_output, disease_name, knowledge):
    user_template = p2_config.get('user_template', '')
    formatted = user_template.format(
        phase1_qa_output = phase1_output,
        disease_name      = disease_name,
        disease_knowledge= knowledge,
    )

    examples = p2_config.get('few_shot_examples', [])
    if examples:
        formatted += "\n\n--- CÁC VÍ DỤ MINH HỌA ---\n"
        for ex_item in examples:
            for _, ex_val in ex_item.items():
                formatted += (
                    f"Ví dụ ({ex_val.get('disease_name', '')}):\n"
                    f"QA Đầu vào:\n"
                    f"{ex_val.get('phase1_qa_output', '').strip()}\n\n"
                    f"JSON Đầu ra:\n"
                    f"{ex_val.get('expected_json', '').strip()}\n\n"
                )
        formatted += (
            f"--> BÂY GIỜ LÀ LƯỢT CỦA BẠN — "
            f"TRÍCH XUẤT JSON CHO BỆNH {disease_name}:"
        )
    return formatted


# ─────────────────────────────────────────────────────────────────
#  Một VLM xử lý 1 ảnh: Phase 1 + Phase 2
# ─────────────────────────────────────────────────────────────────
def run_vlm_single(
    engine: VLMEngine,
    model_label: str,
    image_path: str,
    disease_name: str,
    disease_knowledge: str,
    prompts_config: dict,
) -> dict:
    img_filename = os.path.basename(image_path)
    result = {
        "phase1_raw":      "",
        "phase2_raw":      "",
        "json_extraction": {},
        "success":         False,
    }

    try:
        # ── Phase 1: Quan sát hình thái ──────────────────────
        sys_p1 = prompts_config["phase1_observation_qa"]["system_instruction"]
        usr_p1 = build_p1_prompt(prompts_config["phase1_observation_qa"])

        print(f"\n  [{model_label}/Phase1] Gọi {engine.model_id} — quan sát...")
        p1_res = engine.call_vlm(sys_p1, usr_p1, image_path=image_path)
        result["phase1_raw"] = p1_res

        print(f"\n  [{model_label}/Phase1] ✅ Quan sát thô:")
        print(f"  {'─'*45}")
        for line in p1_res.splitlines()[:8]:
            print(f"  {line}")
        print(f"  {'─'*45}\n")

        save_debug_file(
            f"DEBUG_{model_label}_P1",
            disease_name, img_filename,
            sys_p1, usr_p1, p1_res,
        )

        if p1_res.startswith("LỖI"):
            print(f"  ❌ [{model_label}/Phase1] Lỗi: {p1_res}")
            result["json_extraction"] = {"error": p1_res, "Category": disease_name}
            return result

        # ── Phase 2: Chuẩn hóa JSON ──────────────────────────
        sys_p2 = prompts_config["phase2_json_standardization"]["system_instruction"]
        usr_p2 = build_p2_prompt(
            prompts_config["phase2_json_standardization"],
            p1_res, disease_name, disease_knowledge,
        )

        print(f"  [{model_label}/Phase2] Gọi {engine.model_id} — chuẩn hóa...")
        p2_raw = engine.call_vlm(sys_p2, usr_p2, image_path=None)
        result["phase2_raw"] = p2_raw

        save_debug_file(
            f"DEBUG_{model_label}_P2",
            disease_name, img_filename,
            sys_p2, usr_p2, p2_raw,
        )

        parsed = engine.extract_json(p2_raw)
        je     = parsed.get("JSON_EXTRACTION", parsed)
        result["json_extraction"] = je
        result["success"]         = "error" not in je

        if result["success"]:
            print(f"  [{model_label}/Phase2] ✅ JSON parse OK.")
        else:
            print(f"  [{model_label}/Phase2] ⚠️ Lỗi parse: {je.get('error')}")

    except Exception as e:
        print(f"  ❌ [{model_label}] Lỗi: {e}")
        result["json_extraction"] = {"error": str(e), "Category": disease_name}

    return result


# ─────────────────────────────────────────────────────────────────
#  Xử lý một ảnh — đầy đủ VLM1 + VLM2 + Judge
# ─────────────────────────────────────────────────────────────────
def process_single_image(
    image_path: str,
    disease_txt_path: str,
    disease_folder_name: str,
    settings: dict,
    prompts_config: dict,
    output_dir: str,
):
    # ── Trích tên bệnh từ file .txt ──────────────────────────
    txt_filename = os.path.basename(disease_txt_path)
    disease_name  = re.sub(r'^.*?\-\s*', '', txt_filename)
    disease_name  = re.sub(r'\.txt$', '', disease_name,
                           flags=re.IGNORECASE).strip()
    img_filename  = os.path.basename(image_path)

    print(f"\n{'='*55}")
    print(f"🚀 XỬ LÝ: [{disease_name}] | Ảnh: [{img_filename}]")
    print(f"   Mode : {RUN_MODE}")
    print(f"{'='*55}")

    with open(disease_txt_path, 'r', encoding='utf-8') as f:
        disease_knowledge = f.read().strip()

    vlm1_cfg  = settings["models"]["vlm_1"]
    vlm2_cfg  = settings["models"]["vlm_2"]
    judge_cfg = settings["models"]["judge_llm"]

    vlm1_result = None
    vlm2_result = None

    # ── VLM 1 — Claude Opus 4.6 ────────────────────────────────
    if RUN_MODE in ("BOTH", "VLM1_ONLY"):
        print(f"\n[VLM1] {vlm1_cfg['provider']} / {vlm1_cfg['model_id']}")
        engine1 = VLMEngine()
        engine1.load_model(provider=vlm1_cfg["provider"], model_id=vlm1_cfg["model_id"])
        vlm1_result = run_vlm_single(
            engine1, "VLM1", image_path,
            disease_name, disease_knowledge, prompts_config,
        )
        engine1.flush_memory()

    # ── VLM 2 — GPT-4o (OpenAI) ─────────────────────────────────
    if RUN_MODE in ("BOTH", "VLM2_ONLY"):
        print(f"\n[VLM2] {vlm2_cfg['provider']} / {vlm2_cfg['model_id']}")
        engine2 = VLMEngine()
        engine2.load_model(provider=vlm2_cfg["provider"], model_id=vlm2_cfg["model_id"])
        vlm2_result = run_vlm_single(
            engine2, "VLM2", image_path,
            disease_name, disease_knowledge, prompts_config,
        )
        engine2.flush_memory()

    # ── Judge — Gemma 4 ─────────────────────────────────────────
    if RUN_MODE == "BOTH":
        print(f"\n[Judge] {judge_cfg['model_id']}")
        judge = JudgeEngine(model_id=judge_cfg["model_id"])
        judge.load_model()

        final_json = judge.run(
            vlm1_result["json_extraction"] if vlm1_result else {},
            vlm2_result["json_extraction"] if vlm2_result else {},
            disease_name       = disease_name,
            jaccard_threshold  = settings["pipeline"].get("jaccard_threshold", 0.85),
        )
        judge.flush_memory()

    elif vlm1_result:
        final_json = vlm1_result["json_extraction"]

    elif vlm2_result:
        final_json = vlm2_result["json_extraction"]

    else:
        final_json = {"error": "Không có kết quả VLM nào.", "Category": disease_name}

    # ── Thêm metadata ───────────────────────────────────────────
    if "JSON_EXTRACTION" not in final_json:
        final_json = {"JSON_EXTRACTION": final_json}

    final_json["JSON_EXTRACTION"].setdefault("_metadata", {}).update({
        "source_image": img_filename,
        "run_mode":    RUN_MODE,
        "vlm1_model":  vlm1_cfg.get("model_id") if RUN_MODE != "VLM2_ONLY" else None,
        "vlm2_model":  vlm2_cfg.get("model_id") if RUN_MODE != "VLM1_ONLY" else None,
        "judge_model": judge_cfg.get("model_id") if RUN_MODE == "BOTH" else None,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # ── Lưu JSON cuối cùng ─────────────────────────────────────
    base_name   = os.path.splitext(img_filename)[0]
    save_folder = os.path.join(output_dir, disease_folder_name)
    save_path   = os.path.join(save_folder, f"{base_name}.json")

    save_json_ordered(final_json, save_path)
    print(f"\n✅ Đã lưu → {save_path}")

    return final_json


# ─────────────────────────────────────────────────────────────────
#  Tìm kiếm file kiến thức bệnh trong contents/
# ─────────────────────────────────────────────────────────────────
def find_knowledge_file(contents_dir: str, disease_folder_name: str) -> str | None:
    """
    Tìm file .txt trong contents/ khớp với disease_folder_name.
    Case-insensitive, rough match để chịu được sự khác nhau nhỏ
    về dấu câu / cách viết.
    """
    disease_lower = disease_folder_name.lower().strip()

    if not os.path.isdir(contents_dir):
        return None

    for fname in os.listdir(contents_dir):
        if not fname.lower().endswith('.txt'):
            continue
        # File format: "Toàn bộ nội dung - <DiseaseName>.txt"
        # Lấy phần sau dấu " - " rồi bỏ .txt
        part = re.sub(r'^.*?\-\s*', '', fname)
        part = re.sub(r'\.txt$', '', part, flags=re.IGNORECASE).strip()
        if part.lower() == disease_lower:
            return os.path.join(contents_dir, fname)

    # Fallback: fuzzy match — kiểm tra disease_lower có trong part không
    for fname in os.listdir(contents_dir):
        if not fname.lower().endswith('.txt'):
            continue
        part = re.sub(r'^.*?\-\s*', '', fname)
        part = re.sub(r'\.txt$', '', part, flags=re.IGNORECASE).strip()
        if disease_lower in part.lower() or part.lower() in disease_lower:
            return os.path.join(contents_dir, fname)

    return None


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.time()

    try:
        SETTINGS    = get_settings()
        PROMPTS_CFG = get_prompts()
        print("✅ Đã load settings.yaml & prompts.yaml")
    except Exception as e:
        print(f"❌ Lỗi load config: {e}")
        sys.exit(1)

    # ── Đường dẫn dataset ─────────────────────────────────────
    DATASET_DIR   = os.path.join(project_root, SETTINGS["paths"]["data_raw"])
    CONTENTS_DIR  = os.path.join(DATASET_DIR, "contents")
    IMAGES_DIR    = os.path.join(DATASET_DIR, "images")
    OUTPUT_DIR    = os.path.join(project_root, "Phase_1", "output", "final")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(get_debug_dir(), exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Dataset   : {DATASET_DIR}")
    print(f"  Images    : {IMAGES_DIR}")
    print(f"  Contents  : {CONTENTS_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  Debug dir : {get_debug_dir()}")
    print(f"  Run mode  : {RUN_MODE}")
    print(f"  VLM1      : {SETTINGS['models']['vlm_1']['provider']} / "
          f"{SETTINGS['models']['vlm_1']['model_id']}")
    print(f"  VLM2      : {SETTINGS['models']['vlm_2']['provider']} / "
          f"{SETTINGS['models']['vlm_2']['model_id']}")
    print(f"  Judge     : {SETTINGS['models']['judge_llm']['model_id']}")
    print(f"{'='*50}\n")

    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Không tìm thấy dataset: {IMAGES_DIR}")
        sys.exit(1)

    # ── Duyệt dataset ──────────────────────────────────────────
    total_images   = 0
    skipped_diseases = 0
    disease_folders = sorted([
        d for d in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, d))
    ])

    print(f"Tìm thấy {len(disease_folders)} thư mục bệnh\n")

    for disease_folder in disease_folders:
        disease_img_dir = os.path.join(IMAGES_DIR, disease_folder)

        # Tìm kiến thức bệnh
        knowledge_path = find_knowledge_file(CONTENTS_DIR, disease_folder)
        if not knowledge_path:
            print(f"⚠️ BỎ QUA '{disease_folder}': không tìm thấy .txt trong contents/")
            skipped_diseases += 1
            continue

        # Tìm ảnh
        image_files = sorted([
            f for f in os.listdir(disease_img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not image_files:
            print(f"⚠️ BỎ QUA '{disease_folder}': không có ảnh")
            skipped_diseases += 1
            continue

        print(f"\n📁 [{disease_folder}] — {len(image_files)} ảnh")

        for img_file in image_files:
            img_path = os.path.join(disease_img_dir, img_file)
            try:
                process_single_image(
                    image_path            = img_path,
                    disease_txt_path      = knowledge_path,
                    disease_folder_name   = disease_folder,
                    settings              = SETTINGS,
                    prompts_config        = PROMPTS_CFG,
                    output_dir            = OUTPUT_DIR,
                )
                total_images += 1

            except Exception as e:
                print(f"❌ Lỗi xử lý {img_file}: {e}")

            # Tránh rate-limit API
            time.sleep(1)

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*50}")
    print(f"✅ HOÀN TẤT")
    print(f"   Ảnh đã xử lý : {total_images}")
    print(f"   Thư mục bỏ qua: {skipped_diseases}")
    print(f"   Tổng thời gian: {elapsed:.1f} phút")
    print(f"{'='*50}")