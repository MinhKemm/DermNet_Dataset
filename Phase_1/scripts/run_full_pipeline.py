"""
run_full_pipeline.py — Pipeline điều phối chính sinh dữ liệu VQA Da liễu

Luồng xử lý:
  1. Load Taxonomy + Config
  2. Duyệt từng thư mục bệnh (disease folder)
     a. Phase 1: Anchor Analysis → xác định key_attribute + disease_name_vi
     b. Duyệt từng ảnh trong thư mục bệnh:
        i.  Phase 2: Fact Extraction → trích xuất extracted_facts
        ii. Data Availability Gate → xác định task nào khả dụng
        iii. Phase 3: Sinh VQA cho từng task khả dụng × 4 loại câu hỏi
        iv. Lưu VQA items ra JSON

Chạy:
    cd /Users/binhminh/Desktop/DermNet_Dataset
    python Phase_1/scripts/run_full_pipeline.py
    python Phase_1/scripts/run_full_pipeline.py --disease "Acne vulgaris"
    python Phase_1/scripts/run_full_pipeline.py --dry-run
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# ─── Setup sys.path ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE1_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

# ─── Imports ─────────────────────────────────────────────────────
from taxonomy.taxonomy_loader import (
    get_raw_taxonomy, get_disease_name_vi, get_location_labels,
    get_attribute_pool, get_lesion_vi_labels, get_all_disease_names_vi,
    get_random_distractors,
)
from core.data_gate import get_available_tasks
from tasks.phase1_anchor_analysis.prompt import build_prompt as build_phase1, parse_response as parse_phase1
from tasks.phase2_fact_extraction.prompt import build_prompt as build_phase2, parse_response as parse_phase2

# Task imports
from tasks.task1_location_recognize import prompt_1_1_multi_choice, prompt_1_2_judgement, prompt_1_3_short_answer, prompt_1_4_fill_in_blank
from tasks.task2_attribute_recognize import prompt_2_1_multi_choice, prompt_2_2_judgement, prompt_2_3_short_answer, prompt_2_4_fill_in_blank
from tasks.task3_lesion_recognize import prompt_3_1_multi_choice, prompt_3_2_judgement, prompt_3_3_short_answer, prompt_3_4_fill_in_blank
from tasks.task4_lesion_reasoning import prompt_4_1_multi_choice, prompt_4_2_judgement, prompt_4_3_short_answer, prompt_4_4_fill_in_blank
from tasks.task5_disease_diagnosis import prompt_5_1_multi_choice, prompt_5_2_judgement, prompt_5_3_short_answer, prompt_5_4_fill_in_blank

# ═══════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════
CONTENTS_DIR = PROJECT_ROOT / "dermnet-output" / "contents"
IMAGES_DIR   = PROJECT_ROOT / "dermnet-output" / "images"
OUTPUT_DIR   = PHASE1_DIR / "output" / "vqa"
PROGRESS_FILE = PHASE1_DIR / "output" / "progress.json"


def load_progress() -> dict:
    """Load file tracking tiến độ (disease/image nào đã chạy xong)."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_images": []}


def save_progress(progress: dict):
    """Lưu tiến độ ra file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_knowledge(disease_name_en: str) -> str:
    """Đọc file kiến thức bệnh."""
    txt_path = CONTENTS_DIR / f"Toàn bộ nội dung - {disease_name_en}.txt"
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def generate_vqa_for_image(
    image_path: str,
    extracted_facts: dict,
    taxonomy: dict,
    key_attribute: str,
    available_tasks: list[int],
) -> list[dict]:
    """Sinh tất cả VQA items cho một ảnh dựa trên các task khả dụng.
    
    Returns:
        List[dict]: danh sách VQA items, mỗi item có {image_path, category, type, question, answer}
    """
    vqa_items = []

    # ── Task 1: Location Recognize ────────────────────────────
    if 1 in available_tasks:
        # 1.1 Multi_choice
        item = prompt_1_1_multi_choice.generate_vqa(image_path, extracted_facts, taxonomy)
        if item: vqa_items.append(item)

        # 1.2 Judgement (CÓ + KHÔNG)
        item_yes = prompt_1_2_judgement.generate_vqa(image_path, extracted_facts, taxonomy, target_flag="CÓ")
        if item_yes: vqa_items.append(item_yes)
        item_no = prompt_1_2_judgement.generate_vqa(image_path, extracted_facts, taxonomy, target_flag="KHÔNG")
        if item_no: vqa_items.append(item_no)

        # 1.3 Short_answer
        item = prompt_1_3_short_answer.generate_vqa(image_path, extracted_facts, taxonomy)
        if item: vqa_items.append(item)

        # 1.4 Fill_in_blank
        item = prompt_1_4_fill_in_blank.generate_vqa(image_path, extracted_facts, taxonomy)
        if item: vqa_items.append(item)

    # ── Task 2: Attribute Recognize ───────────────────────────
    if 2 in available_tasks:
        item = prompt_2_1_multi_choice.generate_vqa(image_path, extracted_facts, taxonomy, KEY_ATTRIBUTE=key_attribute)
        if item: vqa_items.append(item)

        item_yes = prompt_2_2_judgement.generate_vqa(image_path, extracted_facts, taxonomy, KEY_ATTRIBUTE=key_attribute, target_flag="CÓ")
        if item_yes: vqa_items.append(item_yes)
        item_no = prompt_2_2_judgement.generate_vqa(image_path, extracted_facts, taxonomy, KEY_ATTRIBUTE=key_attribute, target_flag="KHÔNG")
        if item_no: vqa_items.append(item_no)

        item = prompt_2_3_short_answer.generate_vqa(image_path, extracted_facts, taxonomy, KEY_ATTRIBUTE=key_attribute)
        if item: vqa_items.append(item)

        item = prompt_2_4_fill_in_blank.generate_vqa(image_path, extracted_facts, taxonomy, KEY_ATTRIBUTE=key_attribute)
        if item: vqa_items.append(item)

    # ── Task 3: Lesion Recognize ──────────────────────────────
    if 3 in available_tasks:
        item = prompt_3_1_multi_choice.generate_vqa(image_path, extracted_facts, taxonomy)
        if item: vqa_items.append(item)

        item_yes = prompt_3_2_judgement.generate_vqa(image_path, extracted_facts, taxonomy, target_flag="CÓ")
        if item_yes: vqa_items.append(item_yes)
        item_no = prompt_3_2_judgement.generate_vqa(image_path, extracted_facts, taxonomy, target_flag="KHÔNG")
        if item_no: vqa_items.append(item_no)

        item = prompt_3_3_short_answer.generate_vqa(image_path, extracted_facts, taxonomy)
        if item: vqa_items.append(item)

        item = prompt_3_4_fill_in_blank.generate_vqa(image_path, extracted_facts, taxonomy)
        if item: vqa_items.append(item)

    # ── Task 4: Lesion Reasoning ──────────────────────────────
    if 4 in available_tasks:
        item = prompt_4_1_multi_choice.generate_vqa(image_path, extracted_facts)
        if item: vqa_items.append(item)

        item_yes = prompt_4_2_judgement.generate_vqa(image_path, extracted_facts, target_flag=True)
        if item_yes: vqa_items.append(item_yes)
        item_no = prompt_4_2_judgement.generate_vqa(image_path, extracted_facts, target_flag=False)
        if item_no: vqa_items.append(item_no)

        item = prompt_4_3_short_answer.generate_vqa(image_path, extracted_facts)
        if item: vqa_items.append(item)

        item = prompt_4_4_fill_in_blank.generate_vqa(image_path, extracted_facts)
        if item: vqa_items.append(item)

    # ── Task 5: Disease Diagnosis ─────────────────────────────
    if 5 in available_tasks:
        item = prompt_5_1_multi_choice.generate_vqa(image_path, extracted_facts, taxonomy)
        if item: vqa_items.append(item)

        item_yes = prompt_5_2_judgement.generate_vqa(image_path, extracted_facts, target_flag=True, taxonomy_data=taxonomy)
        if item_yes: vqa_items.append(item_yes)
        item_no = prompt_5_2_judgement.generate_vqa(image_path, extracted_facts, target_flag=False, taxonomy_data=taxonomy)
        if item_no: vqa_items.append(item_no)

        item = prompt_5_3_short_answer.generate_vqa(image_path, extracted_facts)
        if item: vqa_items.append(item)

        item = prompt_5_4_fill_in_blank.generate_vqa(image_path, extracted_facts)
        if item: vqa_items.append(item)

    return vqa_items


def run_pipeline(args):
    """Chạy toàn bộ pipeline sinh VQA."""
    taxonomy = get_raw_taxonomy()
    taxonomy_str = json.dumps(taxonomy, ensure_ascii=False, indent=2)
    progress = load_progress()
    completed = set(progress.get("completed_images", []))

    # Liệt kê tất cả disease folders
    if not IMAGES_DIR.exists():
        print(f"❌ Không tìm thấy thư mục ảnh: {IMAGES_DIR}")
        return

    disease_folders = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])

    if args.disease:
        disease_folders = [d for d in disease_folders if d == args.disease]
        if not disease_folders:
            print(f"❌ Không tìm thấy bệnh: {args.disease}")
            return

    total_diseases = len(disease_folders)
    total_vqa = 0

    print(f"\n{'='*60}")
    print(f"  🏥 DERMNET VQA PIPELINE — PHASE 1")
    print(f"  Số bệnh cần xử lý: {total_diseases}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'PRODUCTION'}")
    print(f"{'='*60}\n")

    for d_idx, disease_name_en in enumerate(disease_folders, 1):
        print(f"\n{'─'*55}")
        print(f"[{d_idx}/{total_diseases}] 📂 {disease_name_en}")
        print(f"{'─'*55}")

        # ── Phase 1: Anchor Analysis ──────────────────────────
        knowledge_text = load_knowledge(disease_name_en)

        # Tra cứu tên tiếng Việt từ taxonomy (không cần gọi LLM)
        disease_name_vi = get_disease_name_vi(disease_name_en)
        if not disease_name_vi:
            disease_name_vi = disease_name_en  # Fallback
            print(f"  ⚠️ Không tìm thấy tên Tiếng Việt, dùng tên gốc: {disease_name_en}")

        # Build Phase 1 prompt (cho LLM xác định key_attribute)
        phase1_prompt = build_phase1(disease_name_en, knowledge_text, taxonomy_str)

        if args.dry_run:
            print(f"  [DRY RUN] Phase 1 prompt: {len(phase1_prompt)} chars")
            # Dùng default key_attribute cho dry run
            key_attribute = "Color"
        else:
            # TODO: Gọi LLM với phase1_prompt để lấy key_attribute
            # Tạm thời dùng default
            key_attribute = "Color"
            print(f"  📋 Phase 1: key_attribute = {key_attribute}, disease_vi = {disease_name_vi}")

        # ── Duyệt ảnh ────────────────────────────────────────
        disease_img_dir = IMAGES_DIR / disease_name_en
        image_files = sorted([
            f for f in disease_img_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        print(f"  📷 Tìm thấy {len(image_files)} ảnh")

        for img_idx, img_path in enumerate(image_files, 1):
            img_key = f"{disease_name_en}/{img_path.name}"

            # Skip nếu đã xử lý
            if img_key in completed and not args.force:
                continue

            print(f"  [{img_idx}/{len(image_files)}] 🖼️  {img_path.name}")

            if args.dry_run:
                print(f"    [DRY RUN] Sẽ sinh VQA cho ảnh này")
                continue

            # ── Phase 2: Fact Extraction ──────────────────────
            phase2_prompt = build_phase2(disease_name_en, disease_name_vi, key_attribute, taxonomy_str)

            # TODO: Gọi LLM với phase2_prompt + ảnh để trích xuất facts
            # Tạm thời dùng mock data — thay thế bằng API call thực tế
            extracted_facts = {
                "location": "Không xác định được trên ảnh",  # Mock - LLM sẽ trả giá trị thật
                "size": "Không xác định được trên ảnh",
                "color": "Đỏ",
                "shape": "Bất quy tắc",
                "quantity": "Nhiều",
                "distribution": "Rải rác",
                "boundary": "Rõ",
                "lesion": "Sẩn",
                "lesion_reasoning": "Tổn thương nhô cao khỏi bề mặt da, mô đặc, đường kính dưới 1cm.",
                "diagnose": disease_name_vi,
            }

            # ── Data Availability Gate ────────────────────────
            available_tasks = get_available_tasks(extracted_facts, key_attribute.lower())
            print(f"    🚦 Tasks khả dụng: {available_tasks}")

            # ── Phase 3: Sinh VQA ─────────────────────────────
            vqa_items = generate_vqa_for_image(
                image_path=str(img_path),
                extracted_facts=extracted_facts,
                taxonomy=taxonomy,
                key_attribute=key_attribute,
                available_tasks=available_tasks,
            )

            # Lọc None
            vqa_items = [item for item in vqa_items if item is not None]

            if vqa_items:
                # Lưu VQA output
                disease_out_dir = OUTPUT_DIR / disease_name_en
                disease_out_dir.mkdir(parents=True, exist_ok=True)
                out_json = disease_out_dir / f"{img_path.stem}_vqa.json"

                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(vqa_items, f, ensure_ascii=False, indent=2)

                total_vqa += len(vqa_items)
                print(f"    ✅ Đã sinh {len(vqa_items)} VQA items → {out_json.name}")

            # Cập nhật tiến độ
            completed.add(img_key)
            progress["completed_images"] = list(completed)
            save_progress(progress)

    # ── Tổng kết ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ HOÀN TẤT PIPELINE")
    print(f"  Tổng VQA items đã sinh: {total_vqa}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"\n  💡 Tiếp theo: chạy 'python Phase_1/scripts/export_tsv.py' để xuất TSV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🏥 Pipeline sinh dữ liệu VQA Da liễu — Phase 1"
    )
    parser.add_argument(
        "--disease", type=str, default=None,
        help="Chỉ chạy cho một bệnh cụ thể (tên folder tiếng Anh)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chạy thử — chỉ hiển thị thông tin, không sinh VQA"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bỏ qua progress, chạy lại tất cả"
    )
    args = parser.parse_args()

    run_pipeline(args)
