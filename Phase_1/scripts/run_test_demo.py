"""
run_test_demo.py — Chạy thử trên 2 case trong test_demo
Dùng để kiểm tra nhanh Phase 1 + Phase 2 trước khi chạy dataset lớn.

Chạy:
    cd /Users/binhminh/Desktop/DermNet_Dataset
    python Phase_1/scripts/run_test_demo.py
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
from Phase_1.utils.json_handler import save_json_ordered


# ─────────────────────────────────────────────────────────────────
#  Cấu hình
# ─────────────────────────────────────────────────────────────────
PROVIDER = "anthropic"                  # "anthropic" | "openai"
MODEL_ID = "claude-sonnet-4-6"   # Claude Opus 4.6


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
    return os.path.join(
        get_debug_dir(),
        f"{prefix}_{safe(disease_name)}__{safe(image_name)}__{ts}.txt"
    )


def save_debug(prefix, disease_name, image_name,
               sys_prompt, user_prompt, raw_text):
    path = build_debug_path(prefix, disease_name, image_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n"
                f"[{prefix}] {PROVIDER.upper()} / {MODEL_ID}\n"
                f"{'='*60}\n"
                f"BỆNH  : {disease_name}\n"
                f"ẢNH   : {image_name}\n"
                f"TIME  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*60}\n"
                f"\n## SYSTEM PROMPT ##\n{sys_prompt}\n"
                f"\n## USER PROMPT ##\n{user_prompt}\n"
                f"\n{'='*60}\n"
                f"## RAW RESPONSE ##\n{'='*60}\n"
                f"{raw_text}\n")
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
#  Main: chạy 1 ảnh demo
# ─────────────────────────────────────────────────────────────────
def run_demo(image_path: str, disease_txt_path: str,
             prompts_config: dict, output_dir: str):

    txt_filename = os.path.basename(disease_txt_path)
    disease_name  = re.sub(r'^.*?\-\s*', '', txt_filename)
    disease_name  = re.sub(r'\.txt$', '', disease_name,
                           flags=re.IGNORECASE).strip()
    img_filename  = os.path.basename(image_path)

    print(f"\n{'='*60}")
    print(f"🚀 TEST DEMO")
    print(f"   Bệnh : {disease_name}")
    print(f"   Ảnh  : {img_filename}")
    print(f"   Model: {PROVIDER.upper()} / {MODEL_ID}")
    print(f"{'='*60}")

    with open(disease_txt_path, 'r', encoding='utf-8') as f:
        disease_knowledge = f.read().strip()

    engine = VLMEngine()
    engine.load_model(provider=PROVIDER, model_id=MODEL_ID)

    try:
        # ═══ PHASE 1: QUAN SÁT HÌNH THÁI ════════════════════════
        sys_p1 = prompts_config["phase1_observation_qa"]["system_instruction"]
        usr_p1 = build_p1_prompt(prompts_config["phase1_observation_qa"])

        print(f"\n[Phase 1] 🔍 Gọi {MODEL_ID} — quan sát hình thái...")
        phase1_raw = engine.call_vlm(sys_p1, usr_p1, image_path=image_path)

        print(f"\n{'─'*50}")
        print(f"[Phase 1] ✅ KẾT QUẢ QUAN SÁT:")
        print(f"{'─'*50}")
        print(phase1_raw)
        print(f"{'─'*50}\n")

        save_debug("TEST_P1", disease_name, img_filename,
                   sys_p1, usr_p1, phase1_raw)

        if phase1_raw.startswith("LỖI"):
            print(f"❌ [Phase 1] Thất bại: {phase1_raw}")
            return

        # ═══ PHASE 2: CHUẨN HÓA JSON ════════════════════════════
        sys_p2 = prompts_config["phase2_json_standardization"]["system_instruction"]
        usr_p2 = build_p2_prompt(
            prompts_config["phase2_json_standardization"],
            phase1_raw, disease_name, disease_knowledge,
        )

        print(f"[Phase 2] 📋 Gọi {MODEL_ID} — chuẩn hóa JSON...")
        phase2_raw = engine.call_vlm(sys_p2, usr_p2, image_path=None)

        print(f"\n{'─'*50}")
        print(f"[Phase 2] ✅ KẾT QUẢ JSON (thô):")
        print(f"{'─'*50}")
        print(phase2_raw[:800])
        print(f"{'─'*50}\n")

        save_debug("TEST_P2", disease_name, img_filename,
                   sys_p2, usr_p2, phase2_raw)

        # Parse JSON
        parsed = engine.extract_json(phase2_raw)
        je     = parsed.get("JSON_EXTRACTION", parsed)

        print(f"\n{'─'*50}")
        print(f"[Phase 2] ✅ JSON ĐÃ PARSE:")
        print(json.dumps(je, indent=2, ensure_ascii=False))
        print(f"{'─'*50}\n")

        if "error" in je:
            print(f"⚠️ JSON parse lỗi: {je.get('error')}")
            return je

        # Metadata + lưu
        je.setdefault("_metadata", {}).update({
            "source_image": img_filename,
            "model":       MODEL_ID,
            "provider":    PROVIDER,
            "phase1_raw":  phase1_raw,
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{disease_name}.json")
        save_json_ordered({"JSON_EXTRACTION": je}, save_path)
        print(f"✅ Đã lưu JSON → {save_path}")

        return {"JSON_EXTRACTION": je}

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

    finally:
        engine.flush_memory()


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start = time.time()

    try:
        SETTINGS    = get_settings()
        PROMPTS_CFG = get_prompts()
        print("✅ Đã load prompts.yaml + settings.yaml")
    except Exception as e:
        print(f"❌ Lỗi load config: {e}")
        sys.exit(1)

    # ── Thư mục test_demo ─────────────────────────────────────
    TEST_DEMO_DIR = os.path.join(project_root, "Phase_1", "test_demo")
    OUTPUT_DIR    = os.path.join(project_root, "Phase_1", "output", "test_demo")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(get_debug_dir(), exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  TEST DEMO — {PROVIDER.upper()} / {MODEL_ID}")
    print(f"  Test dir  : {TEST_DEMO_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  Debug dir : {get_debug_dir()}")
    print(f"{'='*50}\n")

    if not os.path.exists(TEST_DEMO_DIR):
        print(f"❌ Không tìm thấy: {TEST_DEMO_DIR}")
        sys.exit(1)

    # ── Tìm ảnh + .txt trong test_demo ────────────────────────
    test_cases = []
    for root, dirs, files in os.walk(TEST_DEMO_DIR):
        for fname in sorted(files):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, fname)
                txt_name = next(
                    (f for f in files if f.lower().endswith('.txt')),
                    None
                )
                if txt_name:
                    test_cases.append((img_path, os.path.join(root, txt_name)))

    print(f"Tìm thấy {len(test_cases)} case:\n")
    for i, (img, txt) in enumerate(test_cases, 1):
        print(f"  {i}. Ảnh: {os.path.relpath(img, TEST_DEMO_DIR)}")
        print(f"     TXT: {os.path.relpath(txt, TEST_DEMO_DIR)}\n")

    # ── Chạy từng case ──────────────────────────────────────
    for i, (img_path, txt_path) in enumerate(test_cases, 1):
        print(f"\n{'█'*50}")
        print(f"  CASE {i}/{len(test_cases)}")
        print(f"{'█'*50}")
        try:
            run_demo(img_path, txt_path, PROMPTS_CFG, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Lỗi case {i}: {e}")

        if i < len(test_cases):
            print(f"\n⏳ Chờ 10s trước case tiếp theo...")
            time.sleep(10)

    elapsed = (time.time() - start) / 60
    print(f"\n{'='*50}")
    print(f"✅ HOÀN TẤT — {len(test_cases)} cases — {elapsed:.1f} phút")
    print(f"{'='*50}")