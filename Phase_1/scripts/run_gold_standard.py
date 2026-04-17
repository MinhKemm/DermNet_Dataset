"""
run_gold_standard.py — Standalone Gemma Gold Standard Merger

Luồng:
  1. SCAN Claude JSONs trong output/final/ (đã có image_id từ Phase 2)
  2. SCAN GPT-4o JSONs trong gpt4o_outputs/
  3. PAIR từng cặp theo disease + image filename
  4. GỌI Gemma merge → Gold Standard JSON
  5. LƯU:
       _before_merge/IMG_..._claude_BEFORE.json
       _before_merge/IMG_..._gpt4o_BEFORE.json
       _after_merge/ IMG_..._GOLD_STANDARD.json
       (tuỳ chọn) ghi đè _final.json

Output structure:
  output/final/<Disease>/
    <img>_claude.json                  ← Claude gốc (Phase 2)
    <img>_GPT4O_BEFORE_MERGE.json      ← GPT-4o trước merge
    _before_merge/<img>_claude_BEFORE_MERGE.json
    _before_merge/<img>_gpt4o_BEFORE_MERGE.json
    _after_merge/<img>_GOLD_STANDARD.json
    <img>_final.json                  ← Gold Standard (= after merge)

Chạy:
    cd /Users/binhminh/Desktop/DermNet_Dataset
    python Phase_1/scripts/run_gold_standard.py
"""

import os
import sys
import json
import time
import re
import glob

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Phase_1.core.gemma_comparison import GemmaGoldStandardEngine
from Phase_1.utils.json_handler   import (
    save_json_with_ids,
    generate_image_id,
    load_json,
    CANONICAL_KEYS,
)


# ═══════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════
OUTPUT_DIR   = os.path.join(project_root, "Phase_1", "output", "final")
GPT4O_DIR    = os.path.join(project_root, "Phase_1", "gpt4o_outputs")
IMAGES_DIR   = os.path.join(project_root, "dermnet-output", "images")
CONTENTS_DIR = os.path.join(project_root, "dermnet-output", "contents")
WAIT_AFTER  = 10   # giây giữa mỗi ảnh
GEMMA_TEMP  = 0.05


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════
def slugify(name: str) -> str:
    """Chuẩn hóa tên thành dạng so sánh được."""
    return re.sub(r'[^\w]', '', name.strip().lower())


def get_image_key(image_filename: str) -> str:
    """Khóa để map: bỏ extension, slugify."""
    base = os.path.splitext(image_filename)[0]
    return slugify(base)


def get_disease_key(disease_folder: str) -> str:
    """Khóa để map disease."""
    return slugify(disease_folder)


def find_knowledge(disease_folder: str) -> str | None:
    """Tìm file knowledge .txt cho disease_folder."""
    if not os.path.isdir(CONTENTS_DIR):
        return None
    dk = get_disease_key(disease_folder)
    for fname in sorted(os.listdir(CONTENTS_DIR)):
        if not fname.lower().endswith(".txt"):
            continue
        # File format: "Toàn bộ nội dung - <DiseaseName>.txt"
        part = re.sub(r"^.*?\-\s*", "", fname)
        part = re.sub(r"\.txt$", "", part, flags=re.IGNORECASE).strip()
        if get_disease_key(part) == dk:
            return os.path.join(CONTENTS_DIR, fname)
    return None


def load_knowledge(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def find_image_path(disease_folder: str, image_filename: str) -> str | None:
    """Tìm đường dẫn ảnh thực tế."""
    folder_path = os.path.join(IMAGES_DIR, disease_folder)
    if not os.path.isdir(folder_path):
        return None
    # Exact match
    for fname in sorted(os.listdir(folder_path)):
        if fname == image_filename:
            return os.path.join(folder_path, fname)
        # Partial match (bỏ variant suffix)
        base = os.path.splitext(fname)[0]
        img_key = get_image_key(image_filename)
        if get_image_key(fname) == img_key:
            return os.path.join(folder_path, fname)
    return None


def canonicalize(data: dict) -> dict:
    """ÉP canonical field order."""
    inner = data.get("JSON_EXTRACTION", data)
    ordered = {k: inner[k] for k in CANONICAL_KEYS if k in inner}
    return ordered


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Scanner: tìm tất cả Claude JSONs trong output/final/
# ═══════════════════════════════════════════════════════════════
def scan_claude_jsons() -> list[dict]:
    """
    Trả về list entry, mỗi entry:
      {
        "path":          "output/final/Acanthoma fissuratum/acanthoma-fissuratum-01_claude.json",
        "folder":        "Acanthoma fissuratum",
        "image_filename":"acanthoma-fissuratum-01.jpg",
        "data":          <parsed JSON dict>,
        "image_id":      "IMG_...",
        "json_id":       "JSON_IMG_..._claude_...",
      }
    """
    entries = []
    if not os.path.isdir(OUTPUT_DIR):
        return entries

    for root, _, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            # Skip metadata / merged / final
            skip_parts = ["_BEFORE", "_GOLD", "_FINAL", "_before", "_after", "final"]
            if any(s in fname for s in skip_parts):
                continue
            # Ưu tiên file có suffix _claude
            if "_claude.json" not in fname:
                continue

            fpath = os.path.join(root, fname)
            try:
                data = load_json(fpath)
            except Exception:
                continue

            # Trích folder
            rel      = os.path.relpath(fpath, OUTPUT_DIR)
            parts    = rel.split(os.sep)
            folder   = parts[0] if len(parts) > 1 else ""
            # Trích image filename từ json metadata hoặc tên file
            meta     = data.get("JSON_EXTRACTION", {}).get("_metadata", {})
            img_src  = meta.get("source_image", "")
            img_id   = meta.get("image_id", "")
            json_id  = meta.get("json_id", "")

            if not img_src:
                # Fallback: parse từ tên file
                img_src = fname.replace("_claude.json", "").strip()
                if not img_src.endswith(('.jpg', '.png', '.jpeg')):
                    img_src += ".jpg"

            entries.append({
                "path":           fpath,
                "folder":         folder,
                "image_filename": img_src,
                "data":           data,
                "image_id":       img_id,
                "json_id":        json_id,
            })

    print(f"\n[Scanner] Tìm thấy {len(entries)} Claude JSON(s) trong output/final/")
    return entries


# ═══════════════════════════════════════════════════════════════
#  Scanner: tìm GPT-4o JSON trong gpt4o_outputs/
# ═══════════════════════════════════════════════════════════════
def scan_gpt4o_jsons() -> dict[tuple, dict]:
    """
    Trả về dict: (disease_slug, image_key) → {path, data, json_id}
    """
    gpt_map = {}
    if not os.path.isdir(GPT4O_DIR):
        print(f"[Scanner] GPT4O_DIR trống hoặc không tồn tại: {GPT4O_DIR}")
        return gpt_map

    for fname in sorted(os.listdir(GPT4O_DIR)):
        if not fname.lower().endswith(".json"):
            continue
        fpath = os.path.join(GPT4O_DIR, fname)
        try:
            data = load_json(fpath)
        except Exception:
            continue

        inner = data.get("JSON_EXTRACTION", data)
        meta  = inner.get("_metadata", {})
        cat   = meta.get("source_image", "") or meta.get("Category", "")
        dis   = meta.get("disease_name", "")
        json_id = meta.get("json_id", "")

        if not cat and fname:
            # Parse từ tên file: "<image_filename>.json" hoặc "<disease>.json"
            base = os.path.splitext(fname)[0]

        disease_slug = slugify(dis) or slugify(cat)
        image_key   = get_image_key(cat) if cat else get_image_key(base)

        if disease_slug or image_key:
            key = (disease_slug, image_key)
            gpt_map[key] = {
                "path":     fpath,
                "data":     data,
                "json_id":  json_id,
            }

    print(f"[Scanner] Tìm thấy {len(gpt_map)} GPT-4o JSON(s) trong gpt4o_outputs/")
    return gpt_map


# ═══════════════════════════════════════════════════════════════
#  Merger
# ═══════════════════════════════════════════════════════════════
def merge_pair(entry: dict, gpt_data: dict | None, gemma: GemmaGoldStandardEngine) -> dict | None:
    """
    Gọi Gemma merge cho một cặp Claude + GPT-4o JSON.
    Trả về Gold Standard JSON hoặc None nếu lỗi.
    """
    folder      = entry["folder"]
    img_fname   = entry["image_filename"]
    img_id      = entry["image_id"]
    claude_json = entry["data"]
    base_name   = os.path.splitext(img_fname)[0]

    # ── Tìm ảnh thực ─────────────────────────────────────────
    img_path = find_image_path(folder, img_fname)
    if not img_path:
        print(f"[!] Không tìm thấy ảnh: {folder}/{img_fname}")
        return None

    # ── Tìm knowledge ─────────────────────────────────────────
    know_path = find_knowledge(folder)
    knowledge = load_knowledge(know_path)

    # ── Lấy disease name ──────────────────────────────────────
    disease_name = (
        claude_json.get("JSON_EXTRACTION", {})
        .get("_metadata", {})
        .get("disease_name", "")
    )
    if not disease_name:
        disease_name = folder

    # ── Nếu không có GPT-4o → thông báo ─────────────────────
    if gpt_data is None:
        print(f"  ⏭️  BỎ QUA (chưa có GPT-4o): {folder}/{img_fname}")
        return None

    gpt_json = gpt_data["data"]
    gpt_json_id = gpt_data.get("json_id", "")

    # ── Save BEFORE MERGE ────────────────────────────────────
    before_dir = os.path.join(OUTPUT_DIR, folder, "_before_merge")
    os.makedirs(before_dir, exist_ok=True)

    claude_before_path = os.path.join(
        before_dir, f"{base_name}_claude_BEFORE_MERGE.json"
    )
    gpt_before_path = os.path.join(
        before_dir, f"{base_name}_gpt4o_BEFORE_MERGE.json"
    )

    save_json_with_ids(
        claude_json, claude_before_path,
        image_id  = img_id,
        source    = "claude_before_merge",
        extra_meta = {
            "description":  "Claude JSON — trước khi Gemma merge",
            "gpt4o_json_id": gpt_json_id,
        },
    )

    save_json_with_ids(
        gpt_json, gpt_before_path,
        image_id  = img_id,
        source    = "gpt4o_before_merge",
        extra_meta = {
            "description":      "GPT-4o JSON — trước khi Gemma merge",
            "claude_json_id":  entry.get("json_id", ""),
        },
    )

    print(f"  📁 TRƯỚC MERGE → {before_dir}/")

    # ── Gemma Merge ────────────────────────────────────────────
    print(f"  🤖 Gemma đang merge...")

    gold_json = gemma.create_gold_standard(
        image_path        = img_path,
        json_a           = claude_json,
        json_b           = gpt_json,
        disease_name     = disease_name,
        disease_knowledge = knowledge,
        image_id         = img_id,
    )

    # ── Save AFTER MERGE ──────────────────────────────────────
    after_dir  = os.path.join(OUTPUT_DIR, folder, "_after_merge")
    os.makedirs(after_dir, exist_ok=True)

    gold_path = os.path.join(after_dir, f"{base_name}_GOLD_STANDARD.json")
    save_json_with_ids(
        gold_json, gold_path,
        image_id  = img_id,
        source    = "gold",
        extra_meta = {
            "description":       "Gold Standard — sau khi Gemma merge Claude + GPT-4o",
            "claude_json_id":    entry.get("json_id", ""),
            "gpt4o_json_id":     gpt_json_id,
            "before_merge_dir":  os.path.relpath(before_dir, OUTPUT_DIR),
        },
    )

    # ── Save FINAL (= Gold Standard) ─────────────────────────
    final_path = os.path.join(OUTPUT_DIR, folder, f"{base_name}_final.json")
    save_json_with_ids(
        gold_json, final_path,
        image_id  = img_id,
        source    = "gold",
        extra_meta = {
            "description": "Final — Gold Standard JSON",
            "gemma_model": GEMMA4_MODEL_ID,
        },
    )

    print(f"  ✅ GOLD STANDARD → {gold_path}")
    print(f"  ✅ FINAL         → {final_path}")

    return gold_json


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start = time.time()

    print(f"{'='*60}")
    print(f"  GEMMA GOLD STANDARD MERGER")
    print(f"{'='*60}")
    print(f"  Claude JSONs : {OUTPUT_DIR}")
    print(f"  GPT-4o JSONs : {GPT4O_DIR}")
    print(f"  Images       : {IMAGES_DIR}")
    print(f"  Knowledge    : {CONTENTS_DIR}")
    print(f"  Wait between : {WAIT_AFTER}s")
    print(f"{'='*60}\n")

    # ── Scan ──────────────────────────────────────────────────
    claude_entries = scan_claude_jsons()
    gpt4o_map      = scan_gpt4o_jsons()

    if not claude_entries:
        print("❌ Không tìm thấy Claude JSON nào. Chạy Phase 1+2 trước.")
        sys.exit(1)

    if not gpt4o_map:
        print("⚠️  Không tìm thấy GPT-4o JSON nào.")
        print("   Đặt GPT-4o JSON vào:")
        print(f"   → {GPT4O_DIR}/")
        print("   Format tên file: <image_filename>.json  (ví dụ: acanthoma-fissuratum-01.jpg → acanthoma-fissuratum-01.json)")
        print("   Hoặc format: <disease_name>.json")
        print("\n   Các Claude JSON đã sẵn sàng — chạy lại script này sau khi có GPT-4o.")
        sys.exit(0)

    # ── Init Gemma ────────────────────────────────────────────
    gemma = GemmaGoldStandardEngine()
    gemma.load_model()

    # ── Merge từng cặp ────────────────────────────────────────
    merged   = 0
    skipped  = 0
    errors   = 0

    for i, entry in enumerate(claude_entries, 1):
        folder    = entry["folder"]
        img_fname = entry["image_filename"]
        img_key   = get_image_key(img_fname)
        dis_key   = get_disease_key(folder)

        print(f"\n{'─'*55}")
        print(f"[{i}/{len(claude_entries)}] {folder} / {img_fname}")
        print(f"{'─'*55}")

        # Tìm GPT-4o JSON tương ứng
        gpt_data = gpt4o_map.get((dis_key, img_key))
        if not gpt_data:
            # Thử fallback: chỉ theo disease
            for key, val in gpt4o_map.items():
                if key[0] == dis_key:
                    gpt_data = val
                    print(f"  💡 Match theo disease: {key}")
                    break

        if gpt_data:
            try:
                result = merge_pair(entry, gpt_data, gemma)
                if result:
                    merged += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  ❌ Lỗi: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
        else:
            print(f"  ⏭️  BỎ QUA — không tìm thấy GPT-4o JSON cho: {folder}/{img_fname}")
            skipped += 1

        # Đợi giữa các ảnh
        if i < len(claude_entries):
            print(f"\n  ⏳ Đợi {WAIT_AFTER}s...")
            time.sleep(WAIT_AFTER)

    elapsed = (time.time() - start) / 60

    print(f"\n{'='*60}")
    print(f"  HOÀN TẤT")
    print(f"  Đã merge    : {merged}")
    print(f"  Bỏ qua      : {skipped}")
    print(f"  Lỗi         : {errors}")
    print(f"  Thời gian   : {elapsed:.1f} phút")
    print(f"{'='*60}")
    print(f"\n  📁 Output:")
    print(f"     Before : output/final/<Disease>/_before_merge/")
    print(f"     After  : output/final/<Disease>/_after_merge/")
    print(f"     Final  : output/final/<Disease>/<img>_final.json")
