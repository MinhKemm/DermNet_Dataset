"""
json_handler.py — Lưu JSON + ID generation + canonical field order

Chịu trách nhiệm:
  1. Sinh image_id / json_id ngay khi JSON được tạo (Phase 2 metadata)
  2. Ép canonical field order
  3. Lưu đúng format cho cả Claude JSON và Gold Standard JSON
"""

import json
import os
import re
import time
import hashlib


# ═══════════════════════════════════════════════════════════════
#  ID Generators — gọi ngay khi sinh JSON (Phase 2 metadata)
# ═══════════════════════════════════════════════════════════════
def generate_image_id(image_filename: str, disease_folder: str = "") -> str:
    """
    Tạo image_id duy nhất từ tên file ảnh.
    Format: IMG_<disease>_<basename>_<hash4>
    Ví dụ: IMG_Acanthoma_fissuratum_acanthoma_fissuratum_01_a3f2
    """
    safe_name = re.sub(r'[^\w]', '_', image_filename)[:40]
    h         = hashlib.md5(f"{disease_folder}/{image_filename}".encode()).hexdigest()[:4]
    return f"IMG_{safe_name}_{h}"


def generate_json_id(image_id: str, source: str, suffix: str = "") -> str:
    """
    Tạo json_id duy nhất cho mỗi JSON.
    Format: JSON_<image_id>_<source>_<suffix>
    Ví dụ:
      JSON_IMG_Acanthoma_fissuratum_acanthoma_fissuratum_01_a3f2_claude_0421_1200
      JSON_IMG_Acanthoma_fissuratum_acanthoma_fissuratum_01_a3f2_gpt4o_0421_1200
      JSON_IMG_Acanthoma_fissuratum_acanthoma_fissuratum_01_a3f2_gold_0421_1201
    """
    safe_img = re.sub(r'[^\w]', '_', image_id)
    safe_src = re.sub(r'[^\w]', '_', source)
    safe_sfx = re.sub(r'[^\w]', '_', suffix) if suffix else ""
    ts       = time.strftime("%m%d%H%M")
    base     = f"JSON_{safe_img}_{safe_src}_{ts}"
    return base if not safe_sfx else f"{base}_{safe_sfx}"


# ═══════════════════════════════════════════════════════════════
#  Canonical field order
# ═══════════════════════════════════════════════════════════════
CANONICAL_KEYS = [
    "Category",
    "Distribution",    # 1. Vị trí / sắp xếp
    "Lesion_Type",     # 2. Loại tổn thương
    "Colour",          # 3. Màu sắc
    "Shape",           # 4. Hình dạng
    "Characteristics", # 5. Đặc điểm nhận dạng phụ
    "_metadata",
]


def canonicalize_fields(data: dict) -> dict:
    """
    Sắp xếp fields theo CANONICAL_KEYS.
    Bỏ tất cả extra keys (không nằm trong canonical list).
    """
    ordered = {}
    for key in CANONICAL_KEYS:
        if key in data:
            ordered[key] = data[key]
    return ordered


# ═══════════════════════════════════════════════════════════════
#  Save helpers
# ═══════════════════════════════════════════════════════════════
def _ensure_dir(filepath: str):
    d = os.path.dirname(filepath)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_json(data: dict, filepath: str):
    """Lưu Dictionary Python thành file JSON đẹp (Pretty Print)."""
    _ensure_dir(filepath)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[-] [LỖI LƯU FILE] Không thể lưu {filepath}: {e}")


def save_json_ordered(data: dict, filepath: str):
    """
    Lưu JSON với canonical field order.
    """
    ordered = canonicalize_fields(data.get("JSON_EXTRACTION", data))
    wrapped = {"JSON_EXTRACTION": ordered}
    save_json(wrapped, filepath)
    print(f"[+] JSON (Ordered): {filepath}")


def save_json_with_ids(
    data: dict,
    filepath: str,
    image_id: str,
    source: str,
    extra_meta: dict = None,
):
    """
    Lưu JSON đã canonicalize + gắn image_id + json_id vào _metadata.
    Dùng cho cả Claude JSON và Gold Standard JSON.

    Args:
        data:       Dict gốc (có thể đã có "JSON_EXTRACTION" wrapper hoặc chưa)
        filepath:   Đường dẫn lưu file
        image_id:   IMG_... đã sinh
        source:     "claude", "gpt4o", hoặc "gold"
        extra_meta: Dict bổ sung thêm vào _metadata
    """
    _ensure_dir(filepath)

    # Canonicalize inner data
    inner  = canonicalize_fields(data.get("JSON_EXTRACTION", data))
    meta   = dict(inner.get("_metadata", {}))

    # Ghi đè / bổ sung metadata
    meta["image_id"]  = image_id
    meta["json_id"]   = generate_json_id(image_id, source)
    meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if extra_meta:
        meta.update(extra_meta)

    inner["_metadata"] = meta
    wrapped = {"JSON_EXTRACTION": inner}

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(wrapped, f, indent=4, ensure_ascii=False)
        print(f"[+] JSON with IDs: {filepath}")
        print(f"    image_id : {image_id}")
        print(f"    json_id  : {meta['json_id']}")
    except Exception as e:
        print(f"[-] [LỖI LƯU FILE] Không thể lưu {filepath}: {e}")


def load_json(filepath: str) -> dict:
    """Load JSON, chuẩn hóa wrapper JSON_EXTRACTION."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[json_handler] Không tìm thấy file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if "JSON_EXTRACTION" not in data:
        return {"JSON_EXTRACTION": data}
    return data