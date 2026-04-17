"""
run_test_demo.py — Chạy thử trên 2 case trong test_demo
Dùng để kiểm tra nhanh Phase 1 + Phase 2 + Gemma Comparison.

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

from Phase_1.loaders.config_loader   import get_settings, get_prompts
from Phase_1.core.vlm_engine        import VLMEngine
from Phase_1.core.gemma_comparison import GemmaGoldStandardEngine
from Phase_1.loaders.registry       import RegistryManager
from Phase_1.utils.json_handler     import (
    save_json_ordered,
    save_json_with_ids,
    generate_image_id,
    canonicalize_fields,
)


# ─────────────────────────────────────────────────────────────────
#  Cấu hình
# ─────────────────────────────────────────────────────────────────
PROVIDER   = "anthropic"
MODEL_ID   = "claude-sonnet-4-6"
WAIT_BETWEEN_PHASES = 5
WAIT_AFTER_CASE     = 10
RETRY_COUNT         = 3
RETRY_DELAY         = 5
GPT4O_DIR           = os.path.join(project_root, "Phase_1", "gpt4o_outputs")


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
    return os.path.join(get_debug_dir(),
                        f"{prefix}_{safe(disease_name)}__{safe(image_name)}__{ts}.txt")


def save_debug(prefix, disease_name, image_name, sys_prompt, user_prompt, raw_text):
    path = build_debug_path(prefix, disease_name, image_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n"
                f"[{prefix}] {PROVIDER.upper()} / {MODEL_ID}\n"
                f"{'='*60}\n"
                f"BỆNH  : {disease_name}\n"
                f"ẢNH   : {image_name}\n"
                f"TIME  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*60}\n\n"
                f"## SYSTEM ##\n{sys_prompt}\n\n"
                f"## USER ##\n{user_prompt}\n\n"
                f"{'='*60}\n## RAW ##\n{'='*60}\n{raw_text}\n")
    print(f"[DEBUG] Đã lưu → {path}")


def load_gpt4o_json(disease_name: str) -> dict | None:
    if not os.path.exists(GPT4O_DIR):
        return None
    target = disease_name.strip().lower()
    for fname in sorted(os.listdir(GPT4O_DIR)):
        if not fname.lower().endswith(".json"):
            continue
        name_part = os.path.splitext(fname)[0].strip().lower()
        if name_part == target or target in name_part or name_part in target:
            path = os.path.join(GPT4O_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────
#  Retry wrapper
# ─────────────────────────────────────────────────────────────────
def call_with_retry(call_fn, label, max_retries=RETRY_COUNT, retry_delay=RETRY_DELAY):
    for attempt in range(1, max_retries + 1):
        print(f"\n  [{label}] Lần thử #{attempt}/{max_retries}...")
        result = call_fn()
        if result and not result.startswith("LỖI") and result.strip():
            if attempt > 1:
                print(f"  [{label}] ✅ Thành công ở lần #{attempt}")
            return result
        print(f"  [{label}] ⚠️ Thất bại: {str(result)[:80]}")
        if attempt < max_retries:
            print(f"  [{label}] ⏳ Đợi {retry_delay}s trước retry...")
            time.sleep(retry_delay)
    print(f"  [{label}] ❌ Đã thử {max_retries} lần — không thành công.")
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
                user_template += (
                    f"[{img_name}]\n"
                    f"{ex_val.get('expected_output', '').strip()}\n\n"
                )
        user_template += "--> BÂY GIỜ LÀ LƯỢT CỦA BẠN:\n"
    return user_template


def build_p2_prompt(p2_config, phase1_output, disease_name, knowledge):
    user_template = p2_config.get('user_template', '')
    formatted = user_template.format(
        phase1_qa_output = phase1_output,
        disease_name      = disease_name,
        disease_knowledge = knowledge,
    )
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
#  Main
# ─────────────────────────────────────────────────────────────────
def run_demo(image_path, disease_txt_path, prompts_config, output_dir):

    txt_filename = os.path.basename(disease_txt_path)
    disease_name   = re.sub(r'^.*?\-\s*', '', txt_filename)
    disease_name   = re.sub(r'\.txt$', '', disease_name, flags=re.IGNORECASE).strip()
    img_filename   = os.path.basename(image_path)

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
        # ── Phase 1 ─────────────────────────────────────────
        sys_p1 = prompts_config["phase1_observation_qa"]["system_instruction"]
        usr_p1 = build_p1_prompt(prompts_config["phase1_observation_qa"])

        print(f"\n[Phase 1] 🔍 Gọi {MODEL_ID}...")
        phase1_raw = call_with_retry(
            lambda: engine.call_vlm(sys_p1, usr_p1, image_path=image_path),
            "Phase1"
        )

        print(f"\n{'─'*50}")
        print(f"[Phase 1] ✅ KẾT QUẢ:")
        print(f"{'─'*50}")
        print(phase1_raw)
        print(f"{'─'*50}\n")

        save_debug("TEST_P1", disease_name, img_filename, sys_p1, usr_p1, phase1_raw)

        if phase1_raw.startswith("LỖI"):
            print(f"❌ [Phase 1] Thất bại: {phase1_raw}")
            return

        # ⏳ Đợi giữa Phase 1 → Phase 2
        print(f"\n⏳ Đợi {WAIT_BETWEEN_PHASES}s trước Phase 2...")
        time.sleep(WAIT_BETWEEN_PHASES)

        # ── Phase 2 ─────────────────────────────────────────
        sys_p2 = prompts_config["phase2_json_standardization"]["system_instruction"]
        usr_p2 = build_p2_prompt(
            prompts_config["phase2_json_standardization"],
            phase1_raw, disease_name, disease_knowledge,
        )

        print(f"[Phase 2] 📋 Gọi {MODEL_ID} — chuẩn hóa JSON...")
        phase2_raw = call_with_retry(
            lambda: engine.call_vlm(sys_p2, usr_p2, image_path=None),
            "Phase2"
        )

        print(f"\n{'─'*50}")
        print(f"[Phase 2] ✅ JSON thô (500 ký tự đầu):")
        print(f"{'─'*50}")
        print(phase2_raw[:500])
        print(f"{'─'*50}\n")

        save_debug("TEST_P2", disease_name, img_filename, sys_p2, usr_p2, phase2_raw)

        parsed    = engine.extract_json(phase2_raw)
        je_claude = parsed.get("JSON_EXTRACTION", parsed)

        print(f"\n{'─'*50}")
        print(f"[Phase 2] ✅ JSON PARSED:")
        print(json.dumps(je_claude, indent=2, ensure_ascii=False))
        print(f"{'─'*50}\n")

        if "error" in je_claude:
            print(f"⚠️ Parse lỗi: {je_claude.get('error')}")
            return je_claude

        je_claude.setdefault("_metadata", {}).update({
            "source_image": img_filename,
            "model":       MODEL_ID,
            "provider":    PROVIDER,
            "phase1_raw":  phase1_raw,
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        claude_json = {"JSON_EXTRACTION": je_claude}
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{disease_name}_claude.json")
        save_json_ordered(claude_json, save_path)
        print(f"✅ Đã lưu Claude JSON → {save_path}")

        engine.flush_memory()

        # ── Gemma Gold Standard Merge ────────────────────────────
        gpt4o_json = load_gpt4o_json(disease_name)

        if gpt4o_json:
            print(f"\n[GemmaGold] Merge Claude + GPT-4o → Gold Standard...")

            # Save BEFORE MERGE (paper examples)
            before_dir = os.path.join(output_dir, "_before_merge")
            os.makedirs(before_dir, exist_ok=True)
            img_id = generate_image_id(img_filename)

            save_json_with_ids(
                claude_json,
                os.path.join(before_dir, f"{disease_name}_claude_BEFORE_MERGE.json"),
                image_id  = img_id,
                source    = "claude_before_merge",
                extra_meta = {"description": "Claude JSON trước merge"},
            )
            save_json_with_ids(
                gpt4o_json,
                os.path.join(before_dir, f"{disease_name}_gpt4o_BEFORE_MERGE.json"),
                image_id  = img_id,
                source    = "gpt4o_before_merge",
                extra_meta = {"description": "GPT-4o JSON trước merge"},
            )

            # Gemma merge
            gemma = GemmaGoldStandardEngine()
            gemma.load_model()

            gold_json = gemma.create_gold_standard(
                image_path        = image_path,
                json_a           = claude_json,
                json_b           = gpt4o_json,
                disease_name     = disease_name,
                disease_knowledge = disease_knowledge,
                image_id         = img_id,
            )

            # Save AFTER MERGE (paper examples)
            after_dir = os.path.join(output_dir, "_after_merge")
            os.makedirs(after_dir, exist_ok=True)
            gold_path = os.path.join(after_dir, f"{disease_name}_GOLD_STANDARD.json")
            save_json_with_ids(
                gold_json, gold_path,
                image_id  = img_id,
                source    = "gold",
                extra_meta = {
                    "description": "Gold Standard — sau Gemma merge Claude + GPT-4o",
                    "gemma_model": "gemma-3-27b-it",
                },
            )

            # Final = Gold Standard
            final_path = os.path.join(output_dir, f"{disease_name}_final.json")
            save_json_with_ids(
                gold_json, final_path,
                image_id  = img_id,
                source    = "gold",
                extra_meta = {"description": "Final — Gold Standard JSON"},
            )
            print(f"✅ Gold Standard → {gold_path}")
            print(f"✅ Final        → {final_path}")
            gemma.flush_memory()

        else:
            print(f"\n⚠️  Không tìm thấy GPT-4o JSON cho: {disease_name}")
            print(f"   💡 Đặt vào: {GPT4O_DIR}/<tên bệnh>.json")
            img_id = generate_image_id(img_filename)
            final_path = os.path.join(output_dir, f"{disease_name}_final.json")
            save_json_with_ids(
                claude_json, final_path,
                image_id  = img_id,
                source    = "claude",
                extra_meta = {"description": "Final — Claude only (GPT-4o chưa có)"},
            )
            print(f"✅ Dùng Claude làm final → {final_path}")

        return claude_json

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

    TEST_DEMO_DIR = os.path.join(project_root, "Phase_1", "test_demo")
    OUTPUT_DIR    = os.path.join(project_root, "Phase_1", "output", "test_demo")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(get_debug_dir(), exist_ok=True)
    os.makedirs(GPT4O_DIR, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  TEST DEMO — {PROVIDER.upper()} / {MODEL_ID}")
    print(f"  Test dir  : {TEST_DEMO_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  Debug dir : {get_debug_dir()}")
    print(f"  GPT-4o dir: {GPT4O_DIR}")
    print(f"  Wait P1→P2: {WAIT_BETWEEN_PHASES}s")
    print(f"  Wait after case: {WAIT_AFTER_CASE}s")
    print(f"  Retry/phase: {RETRY_COUNT} × {RETRY_DELAY}s")
    print(f"{'='*50}\n")

    if not os.path.exists(TEST_DEMO_DIR):
        print(f"❌ Không tìm thấy: {TEST_DEMO_DIR}")
        sys.exit(1)

    # Tìm ảnh + .txt trong test_demo
    test_cases = []
    for root, _, files in os.walk(TEST_DEMO_DIR):
        for fname in sorted(files):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, fname)
                txt_name = next((f for f in files if f.lower().endswith('.txt')), None)
                if txt_name:
                    test_cases.append((img_path, os.path.join(root, txt_name)))

    print(f"Tìm thấy {len(test_cases)} case:\n")
    for i, (img, txt) in enumerate(test_cases, 1):
        print(f"  {i}. Ảnh: {os.path.relpath(img, TEST_DEMO_DIR)}")
        print(f"     TXT: {os.path.relpath(txt, TEST_DEMO_DIR)}\n")

    for i, (img_path, txt_path) in enumerate(test_cases, 1):
        print(f"\n{'█'*50}")
        print(f"  CASE {i}/{len(test_cases)}")
        print(f"{'█'*50}")
        try:
            run_demo(img_path, txt_path, PROMPTS_CFG, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Lỗi case {i}: {e}")

        if i < len(test_cases):
            print(f"\n⏳ Đợi {WAIT_AFTER_CASE}s trước case tiếp theo...")
            time.sleep(WAIT_AFTER_CASE)

    elapsed = (time.time() - start) / 60
    print(f"\n{'='*50}")
    print(f"✅ HOÀN TẤT — {len(test_cases)} cases — {elapsed:.1f} phút")
    print(f"{'='*50}")