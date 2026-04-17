"""
run_pipeline.py — Phase 1 Full Pipeline trên dermnet-output

Luồng xử lý:
  Phase 1 (Claude) → Phase 2 (Claude) → Gemma Compare vs GPT-4o JSON

Sử dụng master_registry.csv để kiểm soát tiến độ.
Chạy lại sẽ tự động bỏ qua ảnh đã xong.

Dataset:
  dermnet-output/
  ├── images/<DiseaseName>/<image.jpg, *.png>
  └── contents/<Toàn bộ nội dung - DiseaseName.txt>

GPT-4o outputs (bạn đặt vào):
  Phase_1/gpt4o_outputs/<disease_name>.json

Chạy:
    cd /Users/binhminh/Desktop/DermNet_Dataset
    python Phase_1/scripts/run_pipeline.py
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
from Phase_1.core.gemma_comparison import GemmaGoldStandardEngine
from Phase_1.loaders.registry       import RegistryManager, (
    STATUS_PENDING, STATUS_P1_OK, STATUS_P2_OK, STATUS_GEMMA_OK, STATUS_ERROR,
)
from Phase_1.utils.json_handler     import (
    save_json_ordered,
    save_json_with_ids,
    generate_image_id,
    canonicalize_fields,
)


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
                f"[{prefix}] Claude Opus 4.6\n"
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
        phase1_qa_output  = phase1_output,
        disease_name       = disease_name,
        disease_knowledge  = knowledge,
    )

    examples = p2_config.get('few_shot_examples', [])
    if examples:
        formatted += "\n\n--- VÍ DỤ MINH HỌA ---\n"
        for ex_item in examples:
            for _, ex_val in ex_item.items():
                formatted += (
                    f"[{ex_val.get('disease_name', '')}]\n"
                    f"QA Đầu vào:\n"
                    f"{ex_val.get('phase1_qa_output', '').strip()}\n\n"
                    f"JSON Đầu ra:\n"
                    f"{ex_val.get('expected_json', '').strip()}\n\n"
                )
        formatted += f"--> TRÍCH XUẤT JSON CHO BỆNH: {disease_name}\n"

    return formatted


# ─────────────────────────────────────────────────────────────────
#  Tìm GPT-4o JSON trong folder local
# ─────────────────────────────────────────────────────────────────
def load_gpt4o_json(gpt4o_dir: str, disease_name: str) -> dict | None:
    """Tìm JSON của GPT-4o trong gpt4o_outputs/."""
    if not os.path.exists(gpt4o_dir):
        return None

    target = disease_name.strip().lower()
    for fname in sorted(os.listdir(gpt4o_dir)):
        if not fname.lower().endswith(".json"):
            continue
        name_part = os.path.splitext(fname)[0].strip().lower()
        if name_part == target or target in name_part or name_part in target:
            path = os.path.join(gpt4o_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────
#  Xử lý 1 ảnh: Phase 1 + Phase 2 + Gemma Compare
# ─────────────────────────────────────────────────────────────────
def process_single_image(
    image_abs_path: str,
    disease_txt_path: str,
    disease_name: str,
    disease_folder: str,
    settings: dict,
    prompts_config: dict,
    registry: RegistryManager,
    output_dir: str,
    gpt4o_dir: str,
):
    img_rel      = os.path.join(disease_folder, os.path.basename(image_abs_path))
    img_filename = os.path.basename(image_abs_path)

    print(f"\n{'='*55}")
    print(f"🚀 XỬ LÝ: [{disease_name}] | Ảnh: [{img_filename}]")
    print(f"{'='*55}")

    # ── Đọc kiến thức bệnh ──────────────────────────────────────
    with open(disease_txt_path, 'r', encoding='utf-8') as f:
        disease_knowledge = f.read().strip()

    vlm_cfg  = settings["models"]["vlm_1"]
    engine   = VLMEngine()
    engine.load_model(provider=vlm_cfg["provider"], model_id=vlm_cfg["model_id"])

    # ── Phase 1 ───────────────────────────────────────────────
    print(f"\n[Phase1] 🔍 Gọi {vlm_cfg['model_id']}...")
    registry.update_status(img_rel, STATUS_P1_OK, {"phase1_claude_status": "RUNNING"})

    sys_p1 = prompts_config["phase1_observation_qa"]["system_instruction"]
    usr_p1 = build_p1_prompt(prompts_config["phase1_observation_qa"])

    def call_p1():
        return engine.call_vlm(sys_p1, usr_p1, image_path=image_abs_path)

    # Retry logic
    phase1_raw = _call_with_retry(call_p1, "Phase1", max_retries=3, retry_delay=5)

    save_debug_file("P1", disease_name, img_filename, sys_p1, usr_p1, phase1_raw)

    if phase1_raw.startswith("LỖI"):
        print(f"❌ [Phase1] Lỗi: {phase1_raw}")
        registry.update_phase(img_rel, "claude_p1", STATUS_ERROR,
                              raw_text=phase1_raw, error=str(phase1_raw))
        engine.flush_memory()
        return

    print(f"\n[Phase1] ✅ Quan sát OK — 5 dòng đầu:")
    for ln in phase1_raw.splitlines()[:6]:
        print(f"  {ln}")
    registry.update_phase(img_rel, "claude_p1", STATUS_P1_OK, raw_text=phase1_raw)

    # ⏳ Đợi giữa P1 → P2
    print(f"\n⏳ Đợi 5s trước Phase 2...")
    time.sleep(5)

    # ── Phase 2 ───────────────────────────────────────────────
    print(f"\n[Phase2] 📋 Gọi {vlm_cfg['model_id']} — chuẩn hóa JSON...")
    registry.update_status(img_rel, STATUS_P2_OK, {"phase2_claude_status": "RUNNING"})

    sys_p2 = prompts_config["phase2_json_standardization"]["system_instruction"]
    usr_p2 = build_p2_prompt(
        prompts_config["phase2_json_standardization"],
        phase1_raw, disease_name, disease_knowledge,
    )

    def call_p2():
        return engine.call_vlm(sys_p2, usr_p2, image_path=None)

    phase2_raw = _call_with_retry(call_p2, "Phase2", max_retries=3, retry_delay=5)

    save_debug_file("P2", disease_name, img_filename, sys_p2, usr_p2, phase2_raw)

    parsed   = engine.extract_json(phase2_raw)
    je_claude = parsed.get("JSON_EXTRACTION", parsed)

    if "error" in je_claude:
        print(f"⚠️ [Phase2] Parse lỗi: {je_claude.get('error')}")
        registry.update_phase(img_rel, "claude_p2", STATUS_ERROR,
                               raw_text=phase2_raw, error=str(je_claude))
        engine.flush_memory()
        return

    registry.update_phase(img_rel, "claude_p2", STATUS_P2_OK, raw_text=phase2_raw)
    print(f"[Phase2] ✅ Parse OK.")

    # ── Đánh ID ngay từ Phase 2 ─────────────────────────────────
    image_id = generate_image_id(img_filename, disease_folder)

    # Thêm metadata (canonicalize fields trước)
    je_claude = canonicalize_fields(je_claude)
    je_claude.setdefault("_metadata", {}).update({
        "source_image": img_filename,
        "model":       vlm_cfg["model_id"],
        "provider":    vlm_cfg["provider"],
        "phase1_raw":  phase1_raw,
    })

    # Lưu Claude JSON với ID
    claude_json = {"JSON_EXTRACTION": je_claude}
    save_path   = os.path.join(output_dir, disease_folder, f"{os.path.splitext(img_filename)[0]}_claude.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_json_with_ids(
        claude_json, save_path,
        image_id   = image_id,
        source     = "claude",
        extra_meta = {
            "model":    vlm_cfg["model_id"],
            "provider": vlm_cfg["provider"],
            "phase1_raw": phase1_raw,
        },
    )
    print(f"✅ Đã lưu Claude JSON (with IDs) → {save_path}")
    print(f"   image_id: {image_id}")

    engine.flush_memory()

    # ── Gemma Compare vs GPT-4o JSON ──────────────────────────
    gpt4o_json = load_gpt4o_json(gpt4o_dir, disease_name)

    if gpt4o_json:
        print(f"\n[GemmaGold] Merge Claude vs GPT-4o → Gold Standard...")

        # Lưu TRƯỚC MERGE để làm ví dụ paper
        before_dir = os.path.join(output_dir, disease_folder, "_before_merge")
        before_path = os.path.join(before_dir, f"{os.path.splitext(img_filename)[0]}_BEFORE_MERGE.json")
        save_json_with_ids(
            claude_json, before_path,
            image_id   = image_id,
            source     = "claude_before_merge",
            extra_meta = {"description": "JSON Claude trước khi Gemma merge"},
        )
        gpt4o_before_path = os.path.join(before_dir, f"{os.path.splitext(img_filename)[0]}_GPT4O_BEFORE_MERGE.json")
        save_json_with_ids(
            gpt4o_json, gpt4o_before_path,
            image_id   = image_id,
            source     = "gpt4o_before_merge",
            extra_meta = {"description": "JSON GPT-4o trước khi Gemma merge"},
        )

        # ── Gemma Merge ─────────────────────────────────────────
        gemma = GemmaGoldStandardEngine()
        gemma.load_model()

        gold_json = gemma.create_gold_standard(
            image_path        = image_abs_path,
            json_a           = claude_json,     # Claude (source)
            json_b           = gpt4o_json,       # GPT-4o
            disease_name     = disease_name,
            disease_knowledge = disease_knowledge,
            image_id         = image_id,
        )

        # Lưu SAU MERGE
        after_dir = os.path.join(output_dir, disease_folder, "_after_merge")
        after_path = os.path.join(after_dir, f"{os.path.splitext(img_filename)[0]}_GOLD_STANDARD.json")
        save_json_with_ids(
            gold_json, after_path,
            image_id   = image_id,
            source     = "gold",
            extra_meta = {
                "description":     "Gold Standard JSON — sau khi Gemma merge Claude + GPT-4o",
                "claude_json_id":  claude_json.get("JSON_EXTRACTION", {}).get("_metadata", {}).get("json_id", ""),
                "gpt4o_json_id":   gpt4o_json.get("JSON_EXTRACTION", {}).get("_metadata", {}).get("json_id", ""),
                "gemma_model":     "gemma-3-27b-it",
            },
        )
        print(f"✅ Gold Standard → {after_path}")

        # Lưu final (= gold standard)
        final_path = os.path.join(output_dir, disease_folder,
                                  f"{os.path.splitext(img_filename)[0]}_final.json")
        save_json_with_ids(
            gold_json, final_path,
            image_id   = image_id,
            source     = "gold",
            extra_meta = {"description": "Final output — Gold Standard"},
        )
        print(f"✅ Final (Gold Standard) → {final_path}")

        registry.update_phase(
            img_rel, "gemma",
            phase_status=STATUS_GEMMA_OK,
            verdict="MERGED",
            reason="Gemma merge Claude + GPT-4o",
        )

        gemma.flush_memory()

        # ⏳ Đợi sau mỗi case
        print(f"\n⏳ Đợi 10s trước case tiếp theo...")
        time.sleep(10)

        return gold_json

    else:
        # Không có GPT-4o JSON → dùng Claude làm final
        # Vẫn đánh ID nhưng không merge
        final_path = os.path.join(output_dir, disease_folder,
                                  f"{os.path.splitext(img_filename)[0]}_final.json")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        save_json_with_ids(
            claude_json, final_path,
            image_id   = image_id,
            source     = "claude",
            extra_meta = {"description": "Final — Claude only (GPT-4o chưa có)"},
        )
        print(f"✅ Không có GPT-4o JSON — dùng Claude làm final.")
        print(f"   💡 Đặt JSON GPT-4o vào: {gpt4o_dir}/<tên bệnh>.json")
        print(f"   để Gemma merge tự động.")

        registry.update_status(img_rel, STATUS_P2_OK)

        print(f"\n⏳ Đợi 10s trước case tiếp theo...")
        time.sleep(10)
        return claude_json


# ─────────────────────────────────────────────────────────────────
#  Retry wrapper
# ─────────────────────────────────────────────────────────────────
def _call_with_retry(call_fn, label, max_retries=3, retry_delay=5):
    for attempt in range(1, max_retries + 1):
        print(f"\n  [{label}] Lần #{attempt}/{max_retries}...")
        result = call_fn()
        if result and not result.startswith("LỖI") and result.strip():
            if attempt > 1:
                print(f"  [{label}] ✅ Thành công ở lần #{attempt}")
            return result
        print(f"  [{label}] ⚠️ Lần #{attempt} thất bại.")
        if attempt < max_retries:
            print(f"  [{label}] ⏳ Đợi {retry_delay}s trước retry...")
            time.sleep(retry_delay)
    print(f"  [{label}] ❌ Tất cả retry đều thất bại.")
    return result if result else "LỖI: Hết retry"


# ─────────────────────────────────────────────────────────────────
#  Tìm kiến thức bệnh
# ─────────────────────────────────────────────────────────────────
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
    start_time = time.time()

    try:
        SETTINGS    = get_settings()
        PROMPTS_CFG = get_prompts()
        print("✅ Đã load settings.yaml & prompts.yaml")
    except Exception as e:
        print(f"❌ Lỗi load config: {e}")
        sys.exit(1)

    # ── Paths ──────────────────────────────────────────────────
    DATASET_DIR     = os.path.join(project_root, SETTINGS["paths"]["data_raw"])
    CONTENTS_DIR   = os.path.join(DATASET_DIR, "contents")
    IMAGES_DIR     = os.path.join(DATASET_DIR, "images")
    OUTPUT_DIR     = os.path.join(project_root, "Phase_1", "output", "final")
    GPT4O_DIR      = os.path.join(project_root, "Phase_1", "gpt4o_outputs")
    REGISTRY_PATH  = os.path.join(project_root, "Phase_1", "master_registry.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(get_debug_dir(), exist_ok=True)
    os.makedirs(GPT4O_DIR, exist_ok=True)   # tạo sẵn folder cho GPT-4o JSON

    print(f"\n{'='*55}")
    print(f"  Dataset     : {DATASET_DIR}")
    print(f"  Images      : {IMAGES_DIR}")
    print(f"  Contents    : {CONTENTS_DIR}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"  GPT-4o dir  : {GPT4O_DIR}")
    print(f"  Registry    : {REGISTRY_PATH}")
    print(f"  VLM         : {SETTINGS['models']['vlm_1']['provider']} / "
          f"{SETTINGS['models']['vlm_1']['model_id']}")
    print(f"{'='*55}\n")

    # ── Registry ───────────────────────────────────────────────
    registry = RegistryManager(REGISTRY_PATH)
    new_imgs = registry.discover_dataset(IMAGES_DIR, CONTENTS_DIR)

    stats = registry.summary()
    print(f"\n📋 REGISTRY SUMMARY:")
    print(f"  Total   : {stats['total']}")
    print(f"  Pending : {stats['pending']}")
    print(f"  Done    : {stats['completed']}")
    print(f"  Status  : {stats['counts']}\n")

    if stats['pending'] == 0:
        print("✅ Tất cả ảnh đã xử lý xong. Thoát.")
        sys.exit(0)

    # ── Duyệt dataset ─────────────────────────────────────────
    processed  = 0
    errors     = 0

    for entry in registry.get_pending():
        img_rel    = entry["image_path"]
        folder     = entry["disease_folder"]
        disease_name = entry.get("disease_name", folder)

        img_abs = os.path.join(IMAGES_DIR, img_rel)

        if not os.path.exists(img_abs):
            print(f"⚠️ BỎ QUA (không tìm thấy ảnh): {img_abs}")
            errors += 1
            continue

        # Tìm kiến thức bệnh
        knowledge_path = find_knowledge_file(CONTENTS_DIR, folder)
        if not knowledge_path:
            print(f"⚠️ BỎ QUA '{folder}': không tìm thấy .txt trong contents/")
            errors += 1
            continue

        try:
            process_single_image(
                image_abs_path    = img_abs,
                disease_txt_path  = knowledge_path,
                disease_name      = disease_name,
                disease_folder    = folder,
                settings          = SETTINGS,
                prompts_config    = PROMPTS_CFG,
                registry          = registry,
                output_dir        = OUTPUT_DIR,
                gpt4o_dir         = GPT4O_DIR,
            )
            processed += 1

            stats = registry.summary()
            print(f"\n  📋 Tiến độ: {stats['completed']}/{stats['total']} đã xong")

        except Exception as e:
            print(f"❌ Lỗi xử lý {img_rel}: {e}")
            import traceback
            traceback.print_exc()
            registry.update_status(img_rel, STATUS_ERROR,
                                    {"error_log": str(e)})
            errors += 1

        # ⏳ Đợi sau mỗi ảnh
        print(f"\n⏳ Đợi 10s trước ảnh tiếp theo...")
        time.sleep(10)

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*55}")
    print(f"✅ HOÀN TẤT")
    print(f"   Đã xử lý : {processed}")
    print(f"   Lỗi      : {errors}")
    print(f"   Thời gian: {elapsed:.1f} phút")
    print(f"{'='*55}")