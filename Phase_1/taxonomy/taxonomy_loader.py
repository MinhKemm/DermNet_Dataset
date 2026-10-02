"""
taxonomy_loader.py — Load và cung cấp dữ liệu Taxonomy chuẩn hóa

Đọc từ taxonomy_data.json (đã parse từ DermNet_Canonical_Taxonomy.docx)
Cung cấp các hàm truy vấn whitelist cho toàn bộ pipeline.
"""

import json
import os
import random
from pathlib import Path

_CACHE = None


def _load_taxonomy() -> dict:
    """Load taxonomy_data.json, cache lại để không đọc lại nhiều lần."""
    global _CACHE
    if _CACHE is None:
        file_path = Path(__file__).parent.parent / "taxonomy_data.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy taxonomy_data.json tại: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            _CACHE = json.load(f)
    return _CACHE


def get_raw_taxonomy() -> dict:
    """Trả về toàn bộ dict taxonomy gốc."""
    return _load_taxonomy()


# ═══════════════════════════════════════════════════════════════
#  Location (Bảng 1: Cây giải phẫu)
# ═══════════════════════════════════════════════════════════════
def get_location_labels() -> list[str]:
    """Trả về danh sách tất cả vị trí giải phẫu (label ở mức cụ thể nhất)."""
    data = _load_taxonomy()
    seen = set()
    labels = []
    for loc in data.get("locations", []):
        label = loc.get("label", "")
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


# ═══════════════════════════════════════════════════════════════
#  Color (Bảng 2: Màu sắc)
# ═══════════════════════════════════════════════════════════════
def get_color_labels() -> list[str]:
    """Trả về danh sách tất cả màu sắc tổn thương."""
    data = _load_taxonomy()
    return data.get("colors", [])


# ═══════════════════════════════════════════════════════════════
#  Boundary (Bảng 3: Đường bờ)
# ═══════════════════════════════════════════════════════════════
def get_boundary_labels() -> list[str]:
    """Trả về danh sách tất cả loại đường bờ."""
    data = _load_taxonomy()
    return data.get("boundaries", [])


# ═══════════════════════════════════════════════════════════════
#  Shape (Bảng 4: Hình thái)
# ═══════════════════════════════════════════════════════════════
def get_shape_labels() -> list[str]:
    """Trả về danh sách tất cả loại hình dạng."""
    data = _load_taxonomy()
    return data.get("shapes", [])


# ═══════════════════════════════════════════════════════════════
#  Size (Bảng 6: Kích thước)
# ═══════════════════════════════════════════════════════════════
def get_size_labels() -> list[str]:
    """Trả về danh sách tất cả mô tả kích thước."""
    data = _load_taxonomy()
    return data.get("sizes", [])


# ═══════════════════════════════════════════════════════════════
#  Quantity & Distribution (Bảng 5)
# ═══════════════════════════════════════════════════════════════
def get_quantity_distribution_labels() -> list[str]:
    """Trả về danh sách tất cả giá trị số lượng và phân bố."""
    data = _load_taxonomy()
    return data.get("quantities_distributions_raw", [])


# ═══════════════════════════════════════════════════════════════
#  Lesion Types (Bảng 8: Tổn thương cơ bản)
# ═══════════════════════════════════════════════════════════════
def get_lesion_types() -> list[dict]:
    """Trả về danh sách dict tổn thương cơ bản {name_en, name_vi, ...}."""
    data = _load_taxonomy()
    return data.get("lesion_types", [])


def get_lesion_vi_labels() -> list[str]:
    """Trả về danh sách tên tiếng Việt của các tổn thương cơ bản."""
    return [lt["name_vi"] for lt in get_lesion_types() if lt.get("name_vi")]


# ═══════════════════════════════════════════════════════════════
#  Disease Names (Bảng bệnh danh)
# ═══════════════════════════════════════════════════════════════
def get_disease_name_vi(disease_name_en: str) -> str | None:
    """Tra cứu tên tiếng Việt chuẩn hóa từ tên tiếng Anh.
    
    Tìm trong cả Bảng 1 (thuần Việt) và Bảng 2 (thuật ngữ quốc tế).
    """
    data = _load_taxonomy()
    target = disease_name_en.strip().lower()

    for d in data.get("diseases_vietnamese", []):
        if d["name_en"].strip().lower() == target:
            return d["name_vi"]

    for d in data.get("diseases_international", []):
        if d["name_en"].strip().lower() == target:
            return d["name_vi"]

    return None


def get_all_disease_names_vi() -> list[str]:
    """Trả về danh sách TẤT CẢ tên bệnh tiếng Việt (Bảng 1 + Bảng 2)."""
    data = _load_taxonomy()
    names = []
    for d in data.get("diseases_vietnamese", []):
        if d.get("name_vi"):
            names.append(d["name_vi"])
    for d in data.get("diseases_international", []):
        if d.get("name_vi"):
            names.append(d["name_vi"])
    return names


def get_all_disease_names_en() -> list[str]:
    """Trả về danh sách TẤT CẢ tên bệnh tiếng Anh."""
    data = _load_taxonomy()
    names = []
    for d in data.get("diseases_vietnamese", []):
        if d.get("name_en"):
            names.append(d["name_en"])
    for d in data.get("diseases_international", []):
        if d.get("name_en"):
            names.append(d["name_en"])
    return names


# ═══════════════════════════════════════════════════════════════
#  Attribute Pool Lookup (cho Multi_choice & Judgement)
# ═══════════════════════════════════════════════════════════════
def get_attribute_pool(key_attribute: str) -> list[str]:
    """Trả về pool whitelist tương ứng cho từng loại thuộc tính.
    
    Args:
        key_attribute: Một trong Size|Color|Shape|Quantity|Distribution|Boundary
    """
    attr = key_attribute.strip().lower()
    mapping = {
        "color": get_color_labels,
        "shape": get_shape_labels,
        "size": get_size_labels,
        "boundary": get_boundary_labels,
        "quantity": get_quantity_distribution_labels,
        "distribution": get_quantity_distribution_labels,
    }
    fn = mapping.get(attr)
    return fn() if fn else []


# ═══════════════════════════════════════════════════════════════
#  Helper: Lấy đáp án nhiễu ngẫu nhiên
# ═══════════════════════════════════════════════════════════════
def get_random_distractors(correct: str, pool: list[str], n: int = 3) -> list[str]:
    """Lấy n đáp án nhiễu ngẫu nhiên từ pool, loại bỏ đáp án đúng.
    
    Nếu pool không đủ n phần tử, trả về tất cả phần tử còn lại.
    """
    available = [item for item in pool if item.strip() != correct.strip()]
    if len(available) < n:
        return available
    return random.sample(available, n)
