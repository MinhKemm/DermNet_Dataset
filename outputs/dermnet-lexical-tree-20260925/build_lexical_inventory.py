from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\VuLapTrinh2\DermNet_Dataset")
SOURCE = ROOT / "outputs" / "dermnet-vqa-reviewed-20260923"
OUT = ROOT / "outputs" / "dermnet-lexical-tree-20260925"
FILES = {
    "Val_4k": "DermNet_Val_4k.reviewed.tsv",
    "Test_1of3": "DermNet_Test_1of3.reviewed.tsv",
    "Test": "DermNet_Test.reviewed.tsv",
}
EXPECTED_ROWS = {"Val_4k": 2721, "Test_1of3": 7891, "Test": 23681}
NEEDED = {
    "index", "image_path", "question", "answer", "category", "type", "source_index",
    "image_id", "source_disease", "sub_category", "answer_concepts",
}
MCQ_OPTION = re.compile(r"(?m)^\s*([A-D])\.\s*(.*?)\s*$")
QUESTION_SPLIT = re.compile(r"\s*(?:,|;|/|\+|\||\bvà\b)\s*", re.I)


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip().strip(" .;,:\"'“”‘’")
    return text


def key(value: object) -> str:
    return clean_text(value).casefold()


def has_phrase(text: str, phrase: str) -> bool:
    """Match a complete Vietnamese word/phrase, not an accidental substring."""
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def parse_array(value: object) -> list[str]:
    try:
        data = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [clean_text(x) for x in data if clean_text(x)]


def split_compound(value: str) -> list[str]:
    return [x for x in (clean_text(y) for y in QUESTION_SPLIT.split(value)) if x]


# This only normalizes unmistakable wording variants; it does not merge disease
# labels or clinically different morphology labels.
REGION_ALIASES = {
    "vùng mặt": "Mặt", "da mặt": "Mặt", "mặt": "Mặt",
    "vùng má": "Má", "má": "Má", "hai má": "Hai má",
    "vùng thân mình": "Thân mình", "da thân mình": "Thân mình", "da thân": "Thân mình",
    "thân mình": "Thân mình", "thân": "Thân mình",
    "vùng cổ": "Cổ", "cổ": "Cổ", "vùng bụng": "Bụng", "bụng": "Bụng",
    "vùng ngực": "Ngực", "ngực": "Ngực", "vùng nách": "Nách", "nách": "Nách",
    "vùng vai": "Vai", "vai": "Vai", "vùng cằm": "Cằm", "cằm": "Cằm",
    "vùng trán": "Trán", "trán": "Trán", "vùng đỉnh đầu": "Đỉnh đầu", "đỉnh đầu": "Đỉnh đầu",
    "vùng thái dương": "Thái dương", "thái dương": "Thái dương",
    "vùng bẹn": "Bẹn", "bẹn": "Bẹn", "vùng sinh dục": "Sinh dục", "sinh dục": "Sinh dục",
    "mu tay": "Mu bàn tay", "mu bàn tay": "Mu bàn tay",
    "mu chân": "Mu bàn chân", "mu bàn chân": "Mu bàn chân",
    "gan bàn chân": "Lòng bàn chân", "gan chân": "Lòng bàn chân", "lòng bàn chân": "Lòng bàn chân",
    "mí mắt": "Mi mắt", "mi mắt": "Mi mắt", "mí mắt trên": "Mi mắt trên", "mi mắt trên": "Mi mắt trên",
    "mí mắt dưới": "Mi mắt dưới", "mi mắt dưới": "Mi mắt dưới",
    "mí dưới": "Mi mắt dưới", "mí trên": "Mi mắt trên",
    "vùng quanh mắt": "Quanh mắt", "quanh mắt": "Quanh mắt",
    "vùng quanh miệng": "Quanh miệng", "quanh miệng": "Quanh miệng",
    "vùng quanh mũi": "Quanh mũi", "quanh mũi": "Quanh mũi",
    "vùng quanh hậu môn": "Quanh hậu môn", "quanh hậu môn": "Quanh hậu môn",
    "vùng sau tai": "Sau tai", "sau tai": "Sau tai", "vùng trước tai": "Trước tai", "trước tai": "Trước tai",
    "nếp dưới vú": "Nếp dưới vú", "dưới vú": "Nếp dưới vú",
    "vùng đỉnh đầu": "Đỉnh đầu", "vùng chẩm": "Chẩm", "chẩm": "Chẩm",
    "vùng nếp gấp": "Nếp gấp", "nếp gấp da": "Nếp gấp", "nếp gấp": "Nếp gấp",
    "cẳng chân hai bên": "Cẳng chân", "hai cẳng chân": "Cẳng chân",
    "vùng chi": "Chi", "vùng da chi": "Chi", "da chi": "Chi", "vùng chi dưới": "Chi dưới",
    "hai chi dưới": "Chi dưới", "vùng quanh môi": "Quanh môi", "quanh môi": "Quanh môi",
    "quanh núm vú": "Quanh núm vú", "quanh quầng vú": "Quanh quầng vú",
    "vùng rốn": "Quanh rốn", "vùng thân trên": "Thân trên", "vùng thân dưới": "Thân dưới",
    "vùng da thân mình": "Thân mình", "thân mình bên": "Thân mình bên",
    "da đầu vùng chẩm": "Da đầu vùng chẩm", "da đầu chẩm": "Da đầu vùng chẩm",
    "da đầu chẩm bên": "Da đầu vùng chẩm", "vùng da đầu": "Da đầu", "vùng chẩm": "Chẩm",
    "da đầu vùng đỉnh": "Đỉnh đầu", "da đầu đỉnh": "Đỉnh đầu", "đỉnh da đầu": "Đỉnh đầu",
    "quanh đỉnh đầu": "Đỉnh đầu", "vùng gian mày": "Gian mày", "gian mày": "Gian mày",
    "cung mày": "Cung mày", "quanh lông mày": "Quanh lông mày", "quanh mày": "Quanh lông mày",
    "dưới lông mày": "Dưới lông mày", "mi dưới": "Mi mắt dưới", "mi trên": "Mi mắt trên",
    "mi trên trong": "Mi trên trong", "nếp móng gần": "Nếp móng gần", "nếp móng bên": "Nếp móng bên",
    "nếp móng": "Nếp móng", "quanh móng": "Quanh móng", "quanh móng tay": "Quanh móng",
    "đầu móng": "Đầu móng", "đầu xa móng": "Đầu xa móng", "bờ bên móng": "Bờ bên móng",
    "bờ móng bên": "Bờ móng bên", "bản móng": "Bản móng", "phiến móng": "Phiến móng",
    "dưới móng": "Dưới móng", "trung tâm móng": "Trung tâm móng", "móng": "Móng",
    "thân trên": "Thân trên", "thân dưới": "Thân dưới", "mạn sườn": "Mạn sườn",
    "xương ức": "Xương ức", "dưới nếp vú": "Nếp dưới vú", "vùng dưới vú": "Nếp dưới vú",
    "quanh nếp gấp": "Quanh nếp gấp", "nếp gấp tay": "Nếp gấp tay", "nếp gấp chi": "Nếp gấp chi",
    "nếp kẽ da": "Nếp kẽ", "vùng nếp kẽ": "Nếp kẽ", "quanh nếp kẽ": "Nếp kẽ",
    "vùng có lông": "Da có lông", "vùng da có lông": "Da có lông", "vùng da lông": "Da có lông",
    "da lông": "Da có lông", "vùng da trơn": "Da", "vùng da": "Da",
    "vùng tã lót": "Vùng tã", "vùng tã": "Vùng tã", "mỏm cụt": "Mỏm cụt",
    "củng mạc": "Củng mạc", "kết mạc": "Kết mạc", "quanh tai": "Quanh tai",
}


def normalized_region(value: str) -> str:
    original = clean_text(value)
    stripped = re.sub(r"(?i)^phân bố\s+", "", original).strip()
    return REGION_ALIASES.get(key(stripped), stripped)


# Patterns explicitly found in the combined Anatomical_Distribution field.
# Similar cluster phrases stay as separate terms; parent groups show their
# broad relation without silently erasing the source wording.
PATTERN_TERMS: list[tuple[str, str]] = [
    ("đối xứng hai bên", "Laterality_and_symmetry"),
    ("khu trú một bên", "Extent_and_laterality"),
    ("khu trú hai bên", "Extent_and_laterality"),
    ("rải rác hợp lưu", "Extent_and_configuration"),
    ("tụ thành cụm", "Clustered"), ("tụ thành đám", "Clustered"),
    ("tập trung cụm", "Clustered"), ("tụ thành chùm", "Clustered"),
    ("xếp thành vòng", "Annular_or_circular_arrangement"), ("ngoại vi thành vòng", "Annular_or_circular_arrangement"),
    ("trung tâm ảnh", "Spatial_or_configuration_candidate"), ("trung tâm mảng", "Spatial_or_configuration_candidate"),
    ("trung tâm móng", "Spatial_or_configuration_candidate"), ("trung tâm quầng", "Spatial_or_configuration_candidate"),
    ("trung tâm nếp", "Spatial_or_configuration_candidate"), ("trung tâm", "Spatial_or_configuration_candidate"),
    ("trên nền mảng", "Spatial_or_configuration_candidate"),
    ("quanh tổn thương", "Spatial_or_configuration_candidate"), ("quanh mạch", "Spatial_or_configuration_candidate"),
    ("quanh mảng", "Spatial_or_configuration_candidate"), ("theo đường", "Spatial_or_configuration_candidate"),
    ("quanh lỗ xỏ", "Spatial_or_configuration_candidate"), ("liên kết mảng", "Spatial_or_configuration_candidate"),
    ("liên kết", "Spatial_or_configuration_candidate"), ("lân cận", "Spatial_or_configuration_candidate"),
    ("gần nhau", "Spatial_or_configuration_candidate"), ("vòng quanh", "Spatial_or_configuration_candidate"),
    ("ngoại vi", "Spatial_or_configuration_candidate"),
    ("đường giữa", "Spatial_or_configuration_candidate"), ("chồng lấp", "Spatial_or_configuration_candidate"),
    ("vệ tinh", "Spatial_or_configuration_candidate"), ("xung quanh", "Spatial_or_configuration_candidate"),
    ("xếp thành hàng", "Linear_or_row"), ("xếp hàng", "Linear_or_row"),
    ("theo hàng", "Linear_or_row"), ("nằm ngang", "Linear_or_row"),
    ("xếp dọc", "Linear_or_row"), ("dạng dải", "Linear_or_row"), ("dọc trục", "Linear_or_row"),
    ("theo chiều ngang", "Linear_or_row"), ("dọc khe", "Linear_or_row"), ("dọc rãnh", "Linear_or_row"),
    ("xếp thành hàng", "Linear_or_row"),
    ("thành hàng", "Linear_or_row"), ("dọc nếp gấp", "Flexural_or_linear"),
    ("dọc nếp móng", "Periungual_pattern"), ("theo nang lông", "Follicular_pattern"),
    ("quanh nang lông", "Follicular_pattern"), ("vùng da tiếp xúc", "Exposure_or_contact"),
    ("vùng phơi nắng", "Exposure_or_contact"), ("vùng da hở", "Exposure_or_contact"),
    ("da phơi nắng", "Exposure_or_contact"),
    ("vùng tiết bã", "Sebaceous_distribution"),
    ("dạng mạng lưới", "Reticular_or_network_pattern"), ("dạng lưới", "Reticular_or_network_pattern"),
    ("dạng tuyến", "Linear_or_row"), ("dạng đường", "Linear_or_row"), ("tuyến tính", "Linear_or_row"),
    ("đồng tâm", "Annular_or_circular_arrangement"), ("thành vòng", "Annular_or_circular_arrangement"),
    ("mặt trước", "Body_surface_orientation"), ("mặt sau", "Body_surface_orientation"),
    ("mặt bên", "Body_surface_orientation"), ("mặt trong", "Body_surface_orientation"),
    ("mặt ngoài", "Body_surface_orientation"), ("thành mảng", "Extent_and_configuration"),
    ("vòng", "Annular_or_circular_arrangement"),
    ("theo đường rẽ tóc", "Spatial_or_configuration_candidate"), ("theo sẹo", "Spatial_or_configuration_candidate"),
    ("mặt duỗi", "Flexural_or_extensor"), ("mặt gấp", "Flexural_or_extensor"),
    ("lan tỏa nhẹ", "Extent"), ("lan tỏa", "Extent"), ("lan rộng", "Extent"),
    ("toàn thân", "Extent"), ("không đều", "Irregular_or_asymmetric_pattern"),
    ("rời rạc", "Extent_and_configuration"), ("tách biệt", "Extent_and_configuration"),
    ("khu trú", "Extent"), ("rải rác", "Extent"), ("dày đặc", "Density"),
    ("thưa thớt", "Density"), ("thưa", "Density"), ("đơn độc", "Number"),
    ("đơn lẻ", "Number"), ("nhiều tổn thương", "Number"), ("nhiều vị trí", "Number"),
    ("nhiều ngón", "Number"), ("nhiều móng", "Number"), ("nhiều ổ", "Number"),
    ("một tổn thương chính", "Number"), ("nhiều tổn thương", "Number"), ("nhiều nốt", "Number"),
    ("nhiều đám", "Number"), ("nhiều", "Number"), ("đơn ổ", "Number"), ("đa ổ", "Number"),
    ("một ổ", "Number"), ("một móng", "Number"), ("riêng lẻ", "Number"),
    ("đám", "Clustered"), ("nhóm", "Clustered"), ("dải", "Linear_or_row"),
    ("dọc", "Linear_or_row"),
    ("đối xứng", "Laterality_and_symmetry"), ("hai bên", "Laterality_and_symmetry"),
    ("một bên", "Laterality_and_symmetry"), ("không đối xứng", "Laterality_and_symmetry"),
    ("thành cụm", "Clustered"), ("từng cụm", "Clustered"), ("cụm nhỏ", "Clustered"),
    ("cụm thưa", "Clustered"), ("tụ đám", "Clustered"), ("thành đám", "Clustered"),
    ("tụ cụm", "Clustered"), ("cụm", "Clustered"),
    ("hợp lưu", "Extent_and_configuration"), ("theo dải", "Linear_or_row"),
    ("thành dải", "Linear_or_row"), ("lan dọc", "Linear_or_row"), ("dọc chi", "Linear_or_row"),
    ("thành hàng", "Linear_or_row"), ("liên tục", "Extent_and_configuration"),
    ("tập trung", "Extent_and_configuration"), ("lệch bên", "Laterality_and_symmetry"),
]
PATTERN_TERMS.sort(key=lambda x: len(x[0]), reverse=True)


def split_anatomy_and_patterns(value: str) -> list[tuple[str, str, str]]:
    """Return (family, group, term), separating site and distribution labels."""
    output: list[tuple[str, str, str]] = []
    for piece in split_compound(value):
        current = re.sub(r"(?i)^phân bố\s+", "", piece).strip()
        # Lung hilum is an internal thoracic landmark, not a skin/body-surface
        # location. Keep the source value visible for manual review.
        if "rốn phổi" in key(current):
            output.append(("Anatomical_field_other", "Unmapped_or_mixed_value_from_Anatomical_Distribution", current))
            continue
        if key(current) == "trung tâm":
            output.append(("Distribution_pattern", "Spatial_or_configuration_candidate", "Trung tâm"))
            continue
        if key(current) in {"mặt trước", "mặt sau", "mặt bên", "mặt trong", "mặt ngoài"}:
            output.append(("Distribution_pattern", "Body_surface_orientation", current))
            continue
        # Extract pattern tokens embedded in mixed strings such as 'Cẳng chân hai bên'.
        for phrase, group in PATTERN_TERMS:
            if group == "Body_surface_orientation" and key(current) != key(phrase):
                continue
            regex = re.compile(r"(?i)(?<!\w)" + re.escape(phrase) + r"(?!\w)")
            if regex.search(current):
                output.append(("Distribution_pattern", group, phrase[:1].upper() + phrase[1:]))
                current = regex.sub(" ", current)
        current = re.sub(r"(?i)\b(?:và|có đặc điểm|phân bố)\b", " ", current)
        current = re.sub(r"\s+", " ", current).strip(" .,;:-")
        if current and key(current) not in {"có", "không"}:
            region = normalized_region(current)
            if is_oral_region(region):
                output.append(("Oral_region", "Oral cavity", region))
            elif is_anatomical_region(region):
                output.append(("Body_region", "Body region", region))
            else:
                output.append(("Anatomical_field_other", "Unmapped_or_mixed_value_from_Anatomical_Distribution", region))
    return output


def judgement_targets(question: str, category: str) -> list[str]:
    q = re.sub(r"\s+", " ", clean_text(question))
    patterns = [
        r"\bcó phải là\s+(.+?)\s+không\??\s*$",
        r"\bcó màu\s+(.+?)\s+không\??\s*$",
        r"\bcó kiểu phân bố\s+(.+?)\s+không\??\s*$",
        r"\bcó dạng phân bố\s+(.+?)\s+không\??\s*$",
        r"\bcó phân bố\s+(.+?)\s+không\??\s*$",
        r"\bcó vùng da\s+(.+?)\s+không\??\s*$",
        r"\bcó biến đổi thứ phát\s+(.+?)\s+không\??\s*$",
        r"\bcó dấu hiệu\s+(.+?)\s+không\??\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, re.I)
        if match:
            return split_compound(match.group(1))
    quoted = re.search(r"[\"'“‘](.+?)[\"'”’]", q)
    if quoted:
        return split_compound(quoted.group(1))
    for pattern in [
        r"\bcó đặc điểm(?: lâm sàng)?\s+(.+?)\s+không\??\s*$",
        r"\bcó biến đổi thứ phát\s+(.+?)\s+không\??\s*$",
    ]:
        match = re.search(pattern, q, re.I)
        if match:
            return split_compound(match.group(1))
    # Keep extraction conservative; unmatched Judgement rows are audited below.
    return []


def mcq_options(question: str) -> list[str]:
    return [clean_text(m.group(2)) for m in MCQ_OPTION.finditer(question) if clean_text(m.group(2))]


def family_for(category: str, sub_category: str) -> tuple[str, str]:
    if category == "Anatomical_Distribution":
        return "Body_region_or_distribution", "split from combined field"
    if category == "Lesion_Recognition":
        return "Lesion_list", sub_category or "Unspecified"
    if category == "Diagnosis":
        return "Diagnosis_answer_list", "Diagnosis"
    if category.startswith("Attribute_"):
        if category == "Attribute_Color":
            return "Attribute_value_list", "Color"
        if category == "Attribute_Shape":
            return "Attribute_value_list", "Shape"
        return "Attribute_value_list", sub_category or "Clinical_characteristics"
    return "Other", sub_category or category


def category_split(value: str, category: str, sub_category: str, judgement: bool) -> list[tuple[str, str, str]]:
    if category == "Anatomical_Distribution":
        return split_anatomy_and_patterns(value)
    family, group = family_for(category, sub_category)
    if category in {"Lesion_Recognition", "Attribute_Color", "Attribute_Shape", "Attribute_Characteristics"} and judgement:
        terms = split_compound(value)
    else:
        terms = [clean_text(value)]
    return [(family, group, term) for term in terms if term and key(term) not in {"có", "không"}]


def is_oral_region(term: str) -> bool:
    k = key(term)
    # "Môi lớn/bé" are vulvar anatomy, not oral lips.
    if "môi lớn" in k or "môi bé" in k:
        return False
    # These describe external facial skin/perioral sites in this dataset.
    if any(x in k for x in ("khóe miệng", "mép miệng", "quanh miệng", "quanh môi", "ria mép")):
        return False
    if k in {"miệng", "môi", "môi trên", "môi dưới"}:
        return True
    return any(x in k for x in (
        "niêm mạc miệng", "niêm mạc môi", "niêm mạc má", "mặt trong môi", "mặt lưng lưỡi", "lưỡi", "khẩu cái",
        "vòm miệng", "khoang miệng", "lợi", "gingiva", "răng", "hầu họng", "sàn miệng",
    ))


def is_anatomical_region(term: str) -> bool:
    k = key(term)
    if "rốn phổi" in k or "võng mạc" in k or "gai thị" in k:
        return False
    if k in {"đầu", "da", "da có lông", "da lông", "thân mình", "chi", "chi trên", "chi dưới", "niêm mạc"}:
        return True
    anatomical_parts = (
        "da đầu", "đỉnh đầu", "chẩm", "chân tóc", "đường chân tóc", "đường rẽ tóc", "thái dương", "trán", "má", "mặt", "cằm", "hàm", "mày", "mắt", "mắt cá", "mi mắt", "mí mắt", "bờ mi", "lông mi", "mũi", "tai", "râu", "cổ", "gáy", "vai", "cánh tay", "cẳng tay", "khuỷu", "cổ tay", "bàn tay", "ngón tay", "ngón cái", "ngón áp út", "ngón", "mu tay", "lòng bàn tay", "móng tay", "móng", "môi", "miệng", "niêm mạc", "khóe miệng", "mép miệng", "quanh miệng", "quanh môi", "mi trên", "mi dưới", "quanh móng", "đầu ngón", "kẽ ngón", "ngực", "vú", "lưng", "bụng", "rốn", "hông", "nách", "bẹn", "đùi", "gối", "khoeo", "cẳng chân", "cổ chân", "bàn chân", "gót", "gót chân", "gân gót", "ngón chân", "móng chân", "mông", "sinh dục", "âm hộ", "âm đạo", "niệu đạo", "bìu", "dương vật", "quy đầu", "hậu môn", "tầng sinh môn", "nếp gấp", "nếp kẽ", "nếp da", "củng mạc", "kết mạc", "chi", "da chi", "vùng tã", "mạn sườn", "xương ức", "hõm ức", "vùng chậu", "vùng cùng cụt", "thượng đòn", "thân trên", "thân dưới", "thân mình", "thân bên", "tay", "chân", "mỏm cụt", "quanh khớp", "mô cái", "vùng có tóc", "vùng chân tóc", "ria mép", "nếp móng", "bản móng", "đầu móng",
    )
    return any(has_phrase(k, x) for x in anatomical_parts)


def oral_path(term: str) -> list[str]:
    """Return a cautious Level 1-4 path within the separate oral tree."""
    k = key(term)
    if k in {"lợi", "gingiva"}:
        return ["Khoang miệng", "Lợi"]
    if "lợi" in k or "gingiva" in k:
        return ["Khoang miệng", "Lợi", term]
    if "răng" in k:
        return ["Khoang miệng", "Răng", term]
    if "lưỡi" in k:
        return ["Khoang miệng", "Lưỡi", term]
    if "khẩu cái" in k or "vòm miệng" in k:
        return ["Khoang miệng", "Khẩu cái", term]
    if "hầu họng" in k:
        return ["Hầu họng", term]
    if "sàn miệng" in k:
        return ["Khoang miệng", "Sàn miệng", term]
    if "môi" in k:
        return ["Khoang miệng", "Môi", term]
    if "niêm mạc má" in k:
        return ["Khoang miệng", "Niêm mạc miệng", "Niêm mạc má", term]
    if "niêm mạc" in k or "khoang miệng" in k:
        return ["Khoang miệng", "Niêm mạc miệng", term]
    return ["Khoang miệng", "Vị trí trong khoang miệng chưa phân nhóm", term]


def body_path(term: str) -> list[str]:
    k = key(term)
    if "môi lớn" in k or "môi bé" in k:
        return ["Vùng sinh dục-hậu môn", "Âm hộ", term]
    if k == "niêm mạc" or "niêm mạc" in k:
        return ["Niêm mạc", "Vị trí niêm mạc chưa xác định", term]
    if has_phrase(k, "mắt cá"):
        return ["Chi dưới", "Cổ chân", "Mắt cá"]
    if has_phrase(k, "bờ mi"):
        return ["Đầu", "Mặt", "Quanh mắt", "Bờ mi"]
    if has_phrase(k, "chân lông mi") or has_phrase(k, "lông mi"):
        return ["Đầu", "Mặt", "Quanh mắt", "Lông mi"]
    if any(has_phrase(k, x) for x in ("chân tóc", "đường chân tóc", "đường rẽ tóc", "vùng chân tóc")):
        return ["Đầu", "Da đầu", "Đường chân tóc" if "chân tóc" in k else "Đường rẽ tóc"]
    if k in {"thân trên", "vùng thân trên", "thân dưới", "vùng thân dưới", "thân mình trước", "thân mình bên", "thân mình sau", "toàn thân"}:
        return ["Thân mình", term]
    if k in {"quanh xương ức", "xương ức", "hõm ức"}:
        return ["Thân mình", "Ngực", term]
    if k in {"thượng đòn", "vùng thượng đòn"}:
        return ["Thân mình", "Ngực", "Vùng thượng đòn"]
    if k in {"vùng râu", "râu", "vùng ria mép"}:
        return ["Đầu", "Mặt", "Vùng râu"]
    if k in {"hai bàn tay", "mu hai bàn tay"}:
        return ["Chi trên", "Bàn tay", term]
    if k in {"mu ngón tay", "mu đốt ngón tay"}:
        return ["Chi trên", "Bàn tay", "Ngón tay", term]
    if k in {"quanh gối", "vùng gối", "hai gối"}:
        return ["Chi dưới", "Gối", term]
    if k in {"quanh cổ chân", "vùng cổ chân", "mắt cá"}:
        return ["Chi dưới", "Cổ chân", term]
    if k in {"vòm bàn chân", "vùng gót", "vùng gót chân"}:
        return ["Chi dưới", "Bàn chân", "Vòm bàn chân" if "vòm" in k else "Gót chân"]
    if k == "vùng gân gót":
        return ["Chi dưới", "Cổ chân", "Gân gót"]
    if k in {"nếp khuỷu tay", "nếp gấp khuỷu tay", "gấp khuỷu tay"}:
        return ["Chi trên", "Khuỷu tay", "Nếp khuỷu"]
    if k in {"bụng bên", "bụng trên", "vùng bụng bên", "vùng bụng trên"}:
        return ["Thân mình", "Bụng", term]
    if k in {"bề mặt vú", "quanh vú"}:
        return ["Thân mình", "Ngực", "Vú", term]
    if k in {"bao quy đầu", "vành quy đầu"}:
        return ["Vùng sinh dục-hậu môn", "Dương vật", term]
    if k in {"cạnh hậu môn", "vùng hậu môn"}:
        return ["Vùng sinh dục-hậu môn", "Hậu môn", term]
    if k == "quanh niệu đạo":
        return ["Vùng sinh dục-hậu môn", "Niệu đạo", "Quanh niệu đạo"]
    if k == "tiền đình âm đạo":
        return ["Vùng sinh dục-hậu môn", "Âm đạo", "Tiền đình âm đạo"]
    if k in {"vùng chậu", "vùng cùng cụt"}:
        return ["Thân mình", "Vùng chậu", term]
    if k in {"hai tay", "tay"}:
        return ["Chi trên", "Vị trí chi trên chưa xác định", term]
    if k in {"hai chân", "chân"}:
        return ["Chi dưới", "Vị trí chi dưới chưa xác định", term]
    if k in {"nếp da", "nếp"}:
        return ["Da", "Nếp gấp/kẽ", term]
    if k in {"quanh khớp", "khớp"}:
        return ["Vị trí cơ thể chưa phân định", "Quanh khớp" if "khớp" in k else term]
    if k in {"mô cái", "vùng mô cái"}:
        return ["Chi trên", "Bàn tay", "Mô cái"]
    if k in {"móng chân cái", "móng ngón chân cái", "quanh móng chân", "rìa móng chân"}:
        return ["Chi dưới", "Bàn chân", "Ngón chân", "Móng chân"]
    if k in {"móng ngón cái", "móng cái", "ngón áp út", "gốc ngón cái", "mu ngón", "mu đốt ngón", "đốt ngón", "khớp ngón", "bên ngón", "gan ngón", "rìa ngón"}:
        return ["Chi chưa xác định", "Ngón chưa xác định chi", term]
    if k == "mỏm cụt":
        return ["Vị trí cơ thể chưa phân định", "Mỏm cụt"]
    if k in {"củng mạc", "kết mạc"}:
        return ["Đầu", "Mắt", term]
    if k in {"vùng tã", "vùng tã lót"}:
        return ["Thân mình", "Vùng tã"]
    if k in {"vùng có tóc", "vùng tóc"}:
        return ["Đầu", "Da đầu", "Vùng da đầu có tóc"]
    if k in {"vùng chi", "da chi", "vùng da chi", "chi có lông", "da chi lông", "vùng chi dưới", "hai chi dưới"}:
        return ["Chi", "Vị trí chi chưa xác định"]
    if k in {"nếp kẽ", "nếp kẽ da", "quanh nếp kẽ", "quanh nếp gấp", "nếp gấp chi"}:
        return ["Da", "Nếp gấp/kẽ", term]
    if k in {"gian mày", "cung mày", "quanh lông mày", "dưới lông mày", "mi trên trong"}:
        return ["Đầu", "Mặt", "Quanh mắt", term]
    if k in {"mi trên", "mi dưới"}:
        return ["Đầu", "Mặt", "Quanh mắt", "Mi mắt trên" if k == "mi trên" else "Mi mắt dưới"]
    if k in {"quanh miệng", "quanh môi", "ria mép"}:
        return ["Đầu", "Mặt", "Quanh miệng", term]
    if k in {"quanh núm vú", "quanh quầng vú"}:
        return ["Thân mình", "Ngực", "Vú", term]
    if k in {"vùng rốn"}:
        return ["Thân mình", "Bụng", "Quanh rốn"]
    if k in {"thân trên", "thân dưới", "mạn sườn", "xương ức"}:
        return ["Thân mình", term]
    if k in {"da đầu vùng chẩm", "da đầu chẩm", "da đầu chẩm bên"}:
        return ["Đầu", "Da đầu", "Vùng chẩm"]
    if k in {"da đầu vùng đỉnh", "da đầu đỉnh", "đỉnh da đầu", "quanh đỉnh đầu"}:
        return ["Đầu", "Da đầu", "Đỉnh đầu"]
    if k in {"vùng da đầu"}:
        return ["Đầu", "Da đầu"]
    if k in {"đầu ngón", "ngón cái", "ngón", "các ngón", "bên ngón"}:
        return ["Chi", "Ngón chưa xác định chi", term]
    if k in {"nếp móng", "nếp móng gần", "nếp móng bên", "quanh móng", "đầu móng", "đầu xa móng", "bờ bên móng", "bờ móng bên", "bản móng", "phiến móng", "dưới móng", "trung tâm móng", "móng"}:
        return ["Chi chưa xác định", "Móng", "Vị trí móng chưa xác định", term]
    exact = {
        "đầu": ["Đầu"], "da đầu": ["Đầu", "Da đầu"], "đỉnh đầu": ["Đầu", "Da đầu", "Đỉnh đầu"],
        "vùng chẩm": ["Đầu", "Da đầu", "Vùng chẩm"], "chẩm": ["Đầu", "Da đầu", "Vùng chẩm"],
        "đường chân tóc": ["Đầu", "Da đầu", "Đường chân tóc"], "chân tóc": ["Đầu", "Da đầu", "Đường chân tóc"],
        "da đầu trán": ["Đầu", "Da đầu", "Da đầu vùng trán"], "da đầu trước": ["Đầu", "Da đầu", "Da đầu trước"],
        "thái dương": ["Đầu", "Da đầu", "Thái dương"], "trán": ["Đầu", "Mặt", "Trán"],
        "mặt": ["Đầu", "Mặt"], "trung tâm mặt": ["Đầu", "Mặt", "Trung tâm mặt"],
        "má": ["Đầu", "Mặt", "Má"], "hai má": ["Đầu", "Mặt", "Má hai bên"],
        "gò má": ["Đầu", "Mặt", "Gò má"], "má trên": ["Đầu", "Mặt", "Má trên"],
        "má dưới mắt": ["Đầu", "Mặt", "Vùng dưới mắt"], "cằm": ["Đầu", "Mặt", "Cằm"],
        "hàm": ["Đầu", "Mặt", "Hàm"], "đường hàm": ["Đầu", "Mặt", "Đường hàm"],
        "quanh mắt": ["Đầu", "Mặt", "Quanh mắt"], "mi mắt": ["Đầu", "Mặt", "Quanh mắt", "Mi mắt"],
        "mi mắt trên": ["Đầu", "Mặt", "Quanh mắt", "Mi mắt trên"],
        "mi mắt dưới": ["Đầu", "Mặt", "Quanh mắt", "Mi mắt dưới"],
        "mí mắt trên": ["Đầu", "Mặt", "Quanh mắt", "Mi mắt trên"],
        "mí mắt dưới": ["Đầu", "Mặt", "Quanh mắt", "Mi mắt dưới"],
        "góc mắt trong": ["Đầu", "Mặt", "Quanh mắt", "Góc mắt trong"],
        "khóe mắt": ["Đầu", "Mặt", "Quanh mắt", "Khóe mắt"],
        "quanh mũi": ["Đầu", "Mặt", "Mũi", "Quanh mũi"], "mũi": ["Đầu", "Mặt", "Mũi"],
        "sống mũi": ["Đầu", "Mặt", "Mũi", "Sống mũi"], "cánh mũi": ["Đầu", "Mặt", "Mũi", "Cánh mũi"],
        "đầu mũi": ["Đầu", "Mặt", "Mũi", "Đầu mũi"], "chóp mũi": ["Đầu", "Mặt", "Mũi", "Chóp mũi"],
        "cạnh mũi": ["Đầu", "Mặt", "Mũi", "Cạnh mũi"], "rãnh mũi má": ["Đầu", "Mặt", "Quanh miệng", "Rãnh mũi má"],
        "quanh miệng": ["Đầu", "Mặt", "Quanh miệng"], "môi": ["Đầu", "Mặt", "Quanh miệng", "Môi"],
        "môi trên": ["Đầu", "Mặt", "Quanh miệng", "Môi trên"], "môi dưới": ["Đầu", "Mặt", "Quanh miệng", "Môi dưới"],
        "khóe miệng": ["Đầu", "Mặt", "Quanh miệng", "Khóe miệng"], "mép miệng": ["Đầu", "Mặt", "Quanh miệng", "Khóe miệng"],
        "vùng ria mép": ["Đầu", "Mặt", "Quanh miệng", "Vùng ria mép"], "râu": ["Đầu", "Mặt", "Vùng râu"],
        "lông mày": ["Đầu", "Mặt", "Vùng quanh mắt", "Lông mày"], "tai": ["Đầu", "Tai"],
        "vành tai": ["Đầu", "Tai", "Vành tai"], "loa tai": ["Đầu", "Tai", "Vành tai"],
        "dái tai": ["Đầu", "Tai", "Dái tai"], "trước tai": ["Đầu", "Tai", "Trước tai"],
        "sau tai": ["Đầu", "Tai", "Sau tai"], "quanh tai": ["Đầu", "Tai", "Quanh tai"],
        "cổ": ["Cổ"], "cổ trước": ["Cổ", "Cổ trước"], "cổ bên": ["Cổ", "Cổ bên"],
        "cổ sau": ["Cổ", "Cổ sau"], "gáy": ["Cổ", "Gáy"], "cổ gáy": ["Cổ", "Gáy"],
        "thân mình": ["Thân mình"], "thân trên": ["Thân mình", "Thân trên"],
        "ngực": ["Thân mình", "Ngực"], "ngực trên": ["Thân mình", "Ngực", "Ngực trên"],
        "ngực trước": ["Thân mình", "Ngực", "Ngực trước"], "ngực bên": ["Thân mình", "Ngực", "Ngực bên"],
        "vú": ["Thân mình", "Ngực", "Vú"], "quầng vú": ["Thân mình", "Ngực", "Vú", "Quầng vú"],
        "núm vú": ["Thân mình", "Ngực", "Vú", "Núm vú"], "nếp dưới vú": ["Thân mình", "Ngực", "Vú", "Nếp dưới vú"],
        "lưng": ["Thân mình", "Lưng"], "lưng trên": ["Thân mình", "Lưng", "Lưng trên"],
        "lưng dưới": ["Thân mình", "Lưng", "Lưng dưới"], "bụng": ["Thân mình", "Bụng"],
        "bụng dưới": ["Thân mình", "Bụng", "Bụng dưới"], "quanh rốn": ["Thân mình", "Bụng", "Quanh rốn"],
        "hông": ["Thân mình", "Hông"], "nách": ["Thân mình", "Nách"], "hố nách": ["Thân mình", "Nách", "Hố nách"],
        "bẹn": ["Thân mình", "Bẹn"], "nếp bẹn": ["Thân mình", "Bẹn", "Nếp bẹn"],
        "vai": ["Chi trên", "Vai"], "cánh tay": ["Chi trên", "Cánh tay"],
        "cánh tay trên": ["Chi trên", "Cánh tay", "Cánh tay trên"], "mặt trong cánh tay": ["Chi trên", "Cánh tay", "Mặt trong cánh tay"],
        "mặt ngoài cánh tay": ["Chi trên", "Cánh tay", "Mặt ngoài cánh tay"], "cẳng tay": ["Chi trên", "Cẳng tay"],
        "khuỷu tay": ["Chi trên", "Khuỷu tay"], "nếp khuỷu": ["Chi trên", "Khuỷu tay", "Nếp khuỷu"],
        "cổ tay": ["Chi trên", "Cổ tay"], "bàn tay": ["Chi trên", "Bàn tay"],
        "mu bàn tay": ["Chi trên", "Bàn tay", "Mu bàn tay"], "lòng bàn tay": ["Chi trên", "Bàn tay", "Lòng bàn tay"],
        "ngón tay": ["Chi trên", "Bàn tay", "Ngón tay"], "ngón tay cái": ["Chi trên", "Bàn tay", "Ngón tay", "Ngón tay cái"],
        "đầu ngón tay": ["Chi trên", "Bàn tay", "Ngón tay", "Đầu ngón tay"], "kẽ ngón tay": ["Chi trên", "Bàn tay", "Ngón tay", "Kẽ ngón tay"],
        "khớp ngón tay": ["Chi trên", "Bàn tay", "Ngón tay", "Khớp ngón tay"],
        "móng tay": ["Chi trên", "Bàn tay", "Ngón tay", "Móng tay"], "móng ngón tay": ["Chi trên", "Bàn tay", "Ngón tay", "Móng tay"],
        "đùi": ["Chi dưới", "Đùi"], "mặt trong đùi": ["Chi dưới", "Đùi", "Mặt trong đùi"],
        "mặt sau đùi": ["Chi dưới", "Đùi", "Mặt sau đùi"], "gối": ["Chi dưới", "Gối"],
        "nếp khoeo": ["Chi dưới", "Gối", "Nếp khoeo"], "khoeo chân": ["Chi dưới", "Gối", "Nếp khoeo"],
        "cẳng chân": ["Chi dưới", "Cẳng chân"], "cẳng chân dưới": ["Chi dưới", "Cẳng chân", "Cẳng chân dưới"],
        "cổ chân": ["Chi dưới", "Cổ chân"], "gót chân": ["Chi dưới", "Bàn chân", "Gót chân"],
        "bàn chân": ["Chi dưới", "Bàn chân"], "mu bàn chân": ["Chi dưới", "Bàn chân", "Mu bàn chân"],
        "lòng bàn chân": ["Chi dưới", "Bàn chân", "Lòng bàn chân"], "gan bàn chân": ["Chi dưới", "Bàn chân", "Lòng bàn chân"],
        "ngón chân": ["Chi dưới", "Bàn chân", "Ngón chân"], "ngón chân cái": ["Chi dưới", "Bàn chân", "Ngón chân", "Ngón chân cái"],
        "đầu ngón chân": ["Chi dưới", "Bàn chân", "Ngón chân", "Đầu ngón chân"], "kẽ ngón chân": ["Chi dưới", "Bàn chân", "Ngón chân", "Kẽ ngón chân"],
        "móng chân": ["Chi dưới", "Bàn chân", "Ngón chân", "Móng chân"], "móng ngón chân": ["Chi dưới", "Bàn chân", "Ngón chân", "Móng chân"],
        "mông": ["Chi dưới", "Mông"], "nếp liên mông": ["Chi dưới", "Mông", "Nếp liên mông"],
        "quanh hậu môn": ["Vùng sinh dục-hậu môn", "Hậu môn", "Quanh hậu môn"],
        "tầng sinh môn": ["Vùng sinh dục-hậu môn", "Tầng sinh môn"], "âm hộ": ["Vùng sinh dục-hậu môn", "Âm hộ"],
        "bìu": ["Vùng sinh dục-hậu môn", "Bìu"], "dương vật": ["Vùng sinh dục-hậu môn", "Dương vật"],
        "thân dương vật": ["Vùng sinh dục-hậu môn", "Dương vật", "Thân dương vật"],
        "quy đầu": ["Vùng sinh dục-hậu môn", "Dương vật", "Quy đầu"], "rãnh quy đầu": ["Vùng sinh dục-hậu môn", "Dương vật", "Rãnh quy đầu"],
        "niêm mạc miệng": ["Khoang miệng", "Niêm mạc miệng"], "niêm mạc môi": ["Khoang miệng", "Niêm mạc môi"],
        "niêm mạc má": ["Khoang miệng", "Niêm mạc miệng", "Niêm mạc má"], "lưỡi": ["Khoang miệng", "Lưỡi"],
        "mặt lưng lưỡi": ["Khoang miệng", "Lưỡi", "Mặt lưng lưỡi"], "khẩu cái": ["Khoang miệng", "Khẩu cái"],
        "vòm miệng": ["Khoang miệng", "Khẩu cái", "Vòm miệng"], "mặt trong môi": ["Khoang miệng", "Niêm mạc môi", "Mặt trong môi"],
        "niêm mạc": ["Niêm mạc", "Vị trí chưa xác định"], "nếp gấp": ["Da", "Nếp gấp"],
        "da có lông": ["Da", "Vùng da có lông"], "vùng da": ["Da", "Vị trí chưa xác định"],
        "da": ["Da", "Vị trí chưa xác định"], "chi": ["Chi", "Vị trí chi chưa xác định"],
        "chi trên": ["Chi trên"], "chi dưới": ["Chi dưới"], "tay": ["Chi trên", "Vị trí chi trên chưa xác định"],
    }
    if k in exact:
        return exact[k]
    if "móng" in k or "quanh móng" in k or "nếp móng" in k:
        return ["Chi chưa xác định", "Móng", "Vị trí móng chưa xác định", term]
    if any(has_phrase(k, x) for x in ("miệng", "môi", "ria mép")):
        return ["Đầu", "Mặt", "Quanh miệng", term]
    if any(has_phrase(k, x) for x in ("lưỡi", "khẩu cái", "niêm mạc")):
        return ["Chưa phân nhánh giải phẫu", term]
    if any(has_phrase(k, x) for x in ("móng", "quanh móng", "nếp móng")):
        return ["Chi chưa xác định", "Móng", "Vị trí móng chưa xác định", term]
    if any(x in k for x in ("củng mạc", "kết mạc")):
        return ["Đầu", "Mắt", term]
    if any(has_phrase(k, x) for x in ("mũi", "má", "mặt", "cằm", "trán", "thái dương", "mắt", "hàm", "râu", "tai", "môi", "mi trên", "mi dưới", "gian mày", "lông mày", "quanh miệng", "quanh môi", "lông mi")):
        return ["Đầu", "Mặt", "Vị trí mặt chưa phân nhóm", term]
    if any(has_phrase(k, x) for x in ("cẳng chân", "đùi", "gối", "chân", "bàn chân", "ngón chân", "gót", "cổ chân", "mông")):
        return ["Chi dưới", "Vị trí chi dưới chưa phân nhóm", term]
    if any(has_phrase(k, x) for x in ("cẳng tay", "cánh tay", "bàn tay", "ngón tay", "khuỷu", "cổ tay", "vai")):
        return ["Chi trên", "Vị trí chi trên chưa phân nhóm", term]
    if any(has_phrase(k, x) for x in ("cổ", "gáy")):
        return ["Cổ", "Vị trí cổ chưa phân nhóm", term]
    if has_phrase(k, "chi") or has_phrase(k, "tay"):
        return ["Chi", "Vị trí chi chưa xác định", term]
    if "nếp" in k or "kẽ" in k:
        return ["Da", "Nếp gấp/kẽ", term]
    if has_phrase(k, "da") or has_phrase(k, "lông"):
        return ["Da", "Vị trí da chưa phân nhóm", term]
    if any(has_phrase(k, x) for x in ("ngực", "bụng", "lưng", "thân", "nách", "hông")):
        return ["Thân mình", "Vị trí thân mình chưa phân nhóm", term]
    if any(has_phrase(k, x) for x in ("bẹn", "âm hộ", "bìu", "dương vật", "sinh dục", "hậu môn", "quy đầu")):
        return ["Vùng sinh dục-hậu môn", "Vị trí chưa phân nhóm", term]
    return ["Chưa phân nhánh giải phẫu", term]


def lesion_parent(term: str, sub_category: str) -> str:
    k = key(term)
    groups = [
        ("Macule_or_patch", ("dát", "mảng", "ban dát")),
        ("Papule", ("sẩn",)), ("Nodule_or_tumour", ("nốt", "u cục", "khối u", "u mềm", "u nhú")),
        ("Vesicle_or_bulla", ("mụn nước", "bọng nước", "mụn bọng")),
        ("Pustule", ("mụn mủ",)), ("Wheal", ("sẩn phù", "mày đay")),
        ("Comedone", ("mụn đầu", "nhân mụn", "nhân trứng cá")),
        ("Erosion", ("trợt", "vết trợt")), ("Ulcer", ("loét", "vết loét", "ổ loét")),
        ("Scale_or_crust", ("vảy", "vảy tiết", "bong vảy", "đóng vảy")),
        ("Fissure", ("nứt", "khe nứt", "rãnh nứt")), ("Scar", ("sẹo",)),
    ]
    for group, starts in groups:
        if any(k.startswith(x) for x in starts):
            return group
    if sub_category in {"Hair_Morphology", "Hair_and_Surface_Characteristics"}:
        return "Hair_finding"
    if sub_category == "Nail_Morphology":
        return "Nail_finding"
    if sub_category in {"Secondary_Change", "Primary_and_Secondary_Morphology"}:
        return "Secondary_or_mixed_change"
    return "Other_or_unclassified_lesion_label"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for split, filename in FILES.items():
        df = pd.read_csv(SOURCE / filename, sep="\t", dtype=str, keep_default_na=False)
        if len(df) != EXPECTED_ROWS[split]:
            raise ValueError(f"{split}: expected {EXPECTED_ROWS[split]} rows, got {len(df)}")
        missing = NEEDED - set(df.columns)
        if missing:
            raise ValueError(f"{split}: missing required columns {sorted(missing)}")
        if df["quality_tier"].nunique() != 1 or df["quality_tier"].iloc[0] != "rule_checked_visual_not_fully_adjudicated":
            raise ValueError(f"{split}: unexpected quality_tier")
        frames[split] = df

    test_sources = set(frames["Test"]["source_index"])
    test1_sources = set(frames["Test_1of3"]["source_index"])
    if test1_sources - test_sources:
        raise ValueError("Test_1of3 is not a source_index subset of Test")
    overlap = set(frames["Val_4k"]["image_id"]) & set(frames["Test"]["image_id"])
    if overlap:
        raise ValueError(f"Val_4k/Test image overlap: {len(overlap)}")

    # Accumulate memberships, not raw repeated occurrences, so counts mean QA rows.
    acc: dict[tuple[str, str, str], dict] = {}
    unparsed_judgements: list[dict] = []

    def add(family: str, group: str, term: str, split: str, row: pd.Series, role: str) -> None:
        raw_term = clean_text(term)
        term = raw_term
        if not term or key(term) in {"có", "không"}:
            return
        if family in {"Body_region", "Oral_region"}:
            term = normalized_region(term)
        elif family == "Distribution_pattern":
            term = term[:1].upper() + term[1:]
        norm = key(term)
        entry_key = (family, group, norm)
        rec = acc.setdefault(entry_key, {
            "family": family, "group": group, "term": term,
            "raw_forms": Counter(), "source_categories": set(), "source_subcategories": set(),
            "all_rows": defaultdict(set), "gold_rows": defaultdict(set),
            "yes_rows": defaultdict(set), "no_target_rows": defaultdict(set),
            "option_rows": defaultdict(set), "images": defaultdict(set),
        })
        rec["raw_forms"][raw_term] += 1
        rec["source_categories"].add(row["category"])
        if row["sub_category"]:
            rec["source_subcategories"].add(row["sub_category"])
        rid = f"{split}:{row['index']}:{row['source_index']}"
        rec["all_rows"][split].add(rid)
        if row["image_id"]:
            rec["images"][split].add(row["image_id"])
        if role == "gold":
            rec["gold_rows"][split].add(rid)
            rec["yes_rows"][split].add(rid)
        elif role == "judgement_yes":
            rec["yes_rows"][split].add(rid)
        elif role == "judgement_no":
            rec["no_target_rows"][split].add(rid)
        elif role == "option":
            rec["option_rows"][split].add(rid)

    for split, df in frames.items():
        for _, row in df.iterrows():
            category = row["category"]
            sub = row["sub_category"]
            judgement = row["type"] == "Judgement"
            answer_concepts = parse_array(row["answer_concepts"])
            if judgement:
                targets = judgement_targets(row["question"], category)
                if not targets:
                    unparsed_judgements.append({
                        "dataset": split, "index": row["index"], "source_index": row["source_index"],
                        "category": category, "question": row["question"], "answer": row["answer"],
                    })
                for target in targets:
                    for family, group, term in category_split(target, category, sub, True):
                        role = "judgement_yes" if key(row["answer"]) == "có" else "judgement_no"
                        add(family, group, term, split, row, role)
            else:
                for concept in answer_concepts:
                    for family, group, term in category_split(concept, category, sub, False):
                        add(family, group, term, split, row, "gold")

            if row["type"] == "Multi_choice":
                for option in mcq_options(row["question"]):
                    for family, group, term in category_split(option, category, sub, False):
                        add(family, group, term, split, row, "option")

            source_disease = clean_text(row["source_disease"])
            if source_disease:
                add("Source_disease_label", "Source disease", source_disease, split, row, "gold")

    def count(rec: dict, bucket: str, split: str) -> int:
        return len(rec[bucket].get(split, set()))

    def tree_fields(family: str, term: str) -> list[str]:
        if family == "Body_region":
            return body_path(term)
        if family == "Oral_region":
            return oral_path(term)
        if family == "Distribution_pattern":
            group = next((x[1] for x in PATTERN_TERMS if key(x[0]) == key(term)), "Other_pattern")
            return ["Distribution", group, term]
        if family == "Lesion_list":
            return ["Lesion", term]
        if family in {"Diagnosis_answer_list", "Source_disease_label"}:
            return ["Disease", term]
        if family == "Attribute_value_list":
            return ["Attributes", term]
        return [family, term]

    rows: list[dict] = []
    for rec in acc.values():
        family, group, term = rec["family"], rec["group"], rec["term"]
        path = tree_fields(family, term)
        if family in {"Body_region", "Oral_region"}:
            mapping_status = "curated_body_region_path" if not path[0].startswith("Chưa") and not any(
                marker in x.casefold() for x in path for marker in ("chưa xác định", "chưa phân")
            ) else "requires_manual_body_region_mapping"
        elif family == "Anatomical_field_other":
            mapping_status = "unmapped_or_mixed_anatomical_distribution_value"
        else:
            mapping_status = "observed_term_no_disease_translation"
        row = {
            "lexicon_family": family,
            "value_group": group,
            "Level_1": path[0] if len(path) > 0 else "",
            "Level_2": path[1] if len(path) > 1 else "",
            "Level_3": path[2] if len(path) > 2 else "",
            "Level_4": path[3] if len(path) > 3 else "",
            "canonical_term": term,
            "lexicon_parent": lesion_parent(term, group) if family == "Lesion_list" else group,
            "observed_forms": " || ".join(sorted(rec["raw_forms"], key=str.casefold)),
            "source_categories": " || ".join(sorted(rec["source_categories"])),
            "source_subcategories": " || ".join(sorted(rec["source_subcategories"])),
            "val_gold_answer_rows": count(rec, "gold_rows", "Val_4k"),
            "val_yes_support_rows": count(rec, "yes_rows", "Val_4k"),
            "val_no_judgement_target_rows": count(rec, "no_target_rows", "Val_4k"),
            "val_mcq_option_rows": count(rec, "option_rows", "Val_4k"),
            "val_images": len(rec["images"].get("Val_4k", set())),
            "test_gold_answer_rows": count(rec, "gold_rows", "Test"),
            "test_yes_support_rows": count(rec, "yes_rows", "Test"),
            "test_no_judgement_target_rows": count(rec, "no_target_rows", "Test"),
            "test_mcq_option_rows": count(rec, "option_rows", "Test"),
            "test_images": len(rec["images"].get("Test", set())),
            "test_1of3_gold_answer_rows": count(rec, "gold_rows", "Test_1of3"),
            "test_1of3_yes_support_rows": count(rec, "yes_rows", "Test_1of3"),
            "test_1of3_no_judgement_target_rows": count(rec, "no_target_rows", "Test_1of3"),
            "test_1of3_mcq_option_rows": count(rec, "option_rows", "Test_1of3"),
            "test_1of3_images": len(rec["images"].get("Test_1of3", set())),
            "mapping_status": mapping_status,
        }
        rows.append(row)

    inventory = pd.DataFrame(rows)
    sort_cols = ["lexicon_family", "value_group", "canonical_term"]
    inventory = inventory.sort_values(sort_cols, key=lambda s: s.str.casefold()).reset_index(drop=True)
    inventory_path = OUT / "DermNet_lexical_inventory_20260925.tsv"
    inventory.to_csv(inventory_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")

    unmapped = inventory[
        inventory.lexicon_family.isin(["Body_region", "Oral_region"])
        & inventory.mapping_status.eq("requires_manual_body_region_mapping")
    ]
    pd.DataFrame(unparsed_judgements, columns=["dataset", "index", "source_index", "category", "question", "answer"]).to_csv(
        OUT / "DermNet_lexical_unparsed_judgements_20260925.tsv", sep="\t", index=False, encoding="utf-8", lineterminator="\n"
    )

    # Nested representation, retaining counts at term leaves and marking branches that need mapping.
    def build_nested_tree(tree_df: pd.DataFrame) -> dict:
        tree: dict = {}
        for _, item in tree_df.iterrows():
            parent = tree
            last = None
            for level in ("Level_1", "Level_2", "Level_3", "Level_4"):
                label = clean_text(item[level])
                if not label:
                    break
                last = parent.setdefault(label, {"_terms": [], "_children": {}})
                parent = last["_children"]
            if last is not None:
                last["_terms"].append({
                    "term": item.canonical_term,
                    "val_yes_support_rows": int(item.val_yes_support_rows),
                    "test_yes_support_rows": int(item.test_yes_support_rows),
                    "test_no_judgement_target_rows": int(item.test_no_judgement_target_rows),
                    "test_1of3_yes_support_rows": int(item.test_1of3_yes_support_rows),
                    "mapping_status": item.mapping_status,
                })
        return tree

    body_tree = build_nested_tree(inventory[inventory.lexicon_family == "Body_region"])
    oral_tree = build_nested_tree(inventory[inventory.lexicon_family == "Oral_region"])

    split_summary = {split: {"qa_rows": len(df), "images": int(df.image_id.nunique())} for split, df in frames.items()}
    family_counts = inventory.groupby("lexicon_family").size().to_dict()
    family_distinct_terms = inventory.groupby("lexicon_family").canonical_term.nunique().to_dict()
    group_counts = inventory.groupby(["lexicon_family", "value_group"]).size().to_dict()
    qtypes = {split: df.type.value_counts().to_dict() for split, df in frames.items()}
    category_counts = {split: df.category.value_counts().to_dict() for split, df in frames.items()}
    summary = {
        "generated_date": "2026-09-25",
        "source_files": FILES,
        "data_scope": "Val_4k plus Test for unique main counts; Test_1of3 is a subset of Test and is reported as a separate comparison only.",
        "source_row_counts": split_summary,
        "main_unique_qa_rows_val_plus_test": len(frames["Val_4k"]) + len(frames["Test"]),
        "main_unique_images_val_plus_test": int(frames["Val_4k"].image_id.nunique() + frames["Test"].image_id.nunique()),
        "test1_source_indices_missing_from_test": len(test1_sources - test_sources),
        "val_test_image_overlap": len(overlap),
        "category_counts": category_counts,
        "question_type_counts": qtypes,
        "lexicon_entry_counts": {str(k): int(v) for k, v in family_counts.items()},
        "distinct_terms_by_family": {str(k): int(v) for k, v in family_distinct_terms.items()},
        "value_group_entry_counts": {f"{a}::{b}": int(v) for (a, b), v in group_counts.items()},
        "body_region_terms": int((inventory.lexicon_family == "Body_region").sum()),
        "oral_region_terms": int((inventory.lexicon_family == "Oral_region").sum()),
        "body_region_terms_needing_manual_mapping": int(len(unmapped)),
        "unmapped_or_mixed_values_left_in_combined_anatomical_field": int((inventory.lexicon_family == "Anatomical_field_other").sum()),
        "judgement_rows_without_extracted_target": len(unparsed_judgements),
        "review_adjudication_note": "Rule-checked visual data; not physician-adjudicated image by image.",
        "reference_method": "MedLesionVQA ICLR 2026: hierarchical body-region lexical tree; separate lesion, disease, and attribute value lists.",
    }

    inventory_records = inventory.fillna("").to_dict(orient="records")
    json_obj = {
        "metadata": summary,
        "body_region_tree": body_tree,
        "oral_cavity_tree": oral_tree,
        "lexical_inventory": inventory_records,
    }
    json_path = OUT / "DermNet_lexical_tree_20260925.json"
    json_path.write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # A readable Level 1-4 tree for the report, matching the paper's table form.
    body = inventory[inventory.lexicon_family == "Body_region"].copy()
    oral = inventory[inventory.lexicon_family == "Oral_region"].copy()
    body = body.sort_values(["Level_1", "Level_2", "Level_3", "Level_4", "canonical_term"], key=lambda s: s.str.casefold())
    lines = [
        "# DermNet VQA: cây thuật ngữ và danh sách nhãn",
        "",
        "Ngày tạo: 2026-09-25",
        "",
        "## Tóm tắt",
        "",
        f"Lập từ {len(frames['Val_4k']):,} dòng Val_4k và {len(frames['Test']):,} dòng Test: tổng {summary['main_unique_qa_rows_val_plus_test']:,} QA, {summary['main_unique_images_val_plus_test']:,} ảnh không giao nhau. Test_1of3 có {len(frames['Test_1of3']):,} dòng và là tập con của Test; không cộng thêm vào tổng chính.",
        f"Từ điển máy đọc được có {len(inventory):,} bản ghi theo tổ hợp nhóm/nhãn, gồm {sum(family_distinct_terms.values()):,} chuỗi nhãn riêng biệt trong từng nhóm chính; cùng một nhãn có thể xuất hiện ở nhiều nhóm nguồn. Trong đó có {family_counts.get('Body_region', 0):,} bản ghi vị trí cơ thể, {family_counts.get('Oral_region', 0):,} bản ghi khoang miệng/niêm mạc, {family_counts.get('Distribution_pattern', 0):,} bản ghi kiểu phân bố, {family_counts.get('Lesion_list', 0):,} bản ghi loại tổn thương, {family_counts.get('Diagnosis_answer_list', 0):,} bản ghi đáp án bệnh danh, {family_counts.get('Source_disease_label', 0):,} bản ghi bệnh danh nguồn và {family_counts.get('Attribute_value_list', 0):,} bản ghi thuộc tính.",
        "",
        "## Cách áp dụng cấu trúc bài báo",
        "",
        "MedLesionVQA trình bày vị trí cơ thể theo cây nhiều cấp, tách cây khoang miệng, rồi liệt kê riêng tổn thương, bệnh và giá trị thuộc tính. Bản này áp dụng cùng cách tổ chức nhưng chỉ đưa vào thuật ngữ có trong các TSV đã rà. Trường Anatomical_Distribution của DermNet đang gộp vị trí, kiểu phân bố và một số loại thông tin khác; chỉ các cụm nhận diện an toàn mới được tách. Giá trị chưa phân loại chắc chắn được giữ trong nhánh riêng để rà soát, không ép vào cây vị trí.",
        "",
        "Nguồn phương pháp: [MedLesionVQA, ICLR 2026, phần Annotation Protocol và Supplementary Tables 3–7](https://proceedings.iclr.cc/paper_files/paper/2026/file/d82c24b7a4237aa4283b38e12047dc38-Paper-Conference.pdf). PDF người dùng gửi: `15645_MedLesionVQA_A_Multimoda (2).pdf`.",
        "",
        "## Cây vị trí cơ thể Level 1–4",
        "",
        "Chỉ các nhãn quan sát được trong đầu vào mới được đưa vào cây. Bảng dùng bốn cột phân cấp như bài báo. Các nhãn có vị trí rộng hoặc chưa xác định được chi/vùng cụ thể được đánh dấu `requires_manual_body_region_mapping` trong TSV.",
        "",
        "| Level 1 | Level 2 | Level 3 | Level 4 | Nhãn trong dữ liệu | QA dương Val | QA dương Test | Judgement phủ định Test |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for _, item in body.iterrows():
        levels = [clean_text(item[x]) for x in ("Level_1", "Level_2", "Level_3", "Level_4")]
        cells = [*levels, clean_text(item.canonical_term), str(int(item.val_yes_support_rows)),
                 str(int(item.test_yes_support_rows)), str(int(item.test_no_judgement_target_rows))]
        lines.append("| " + " | ".join(x.replace("|", "\\|") for x in cells) + " |")
    lines += [
        "",
        "### Cây khoang miệng/niêm mạc",
        "",
        "Bài báo tách riêng cây khoang miệng; các nhãn niêm mạc trong DermNet cũng được để riêng.",
        "",
        "| Level 1 | Level 2 | Level 3 | Level 4 | Nhãn trong dữ liệu | QA dương Val | QA dương Test | Judgement phủ định Test |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    oral = oral.sort_values(["Level_1", "Level_2", "Level_3", "Level_4", "canonical_term"], key=lambda s: s.str.casefold())
    for _, item in oral.iterrows():
        levels = [clean_text(item[x]) for x in ("Level_1", "Level_2", "Level_3", "Level_4")]
        cells = [*levels, clean_text(item.canonical_term), str(int(item.val_yes_support_rows)),
                 str(int(item.test_yes_support_rows)), str(int(item.test_no_judgement_target_rows))]
        lines.append("| " + " | ".join(x.replace("|", "\\|") for x in cells) + " |")
    lines += [
        "",
        "## Danh sách giá trị theo nhóm",
        "",
        "Số liệu tần suất trong file TSV đếm số dòng QA có nhãn đúng hoặc lựa chọn trong câu hỏi; đáp án `Judgement=Không` được ghi riêng là nhãn được hỏi phủ định, không tính là bằng chứng dương.",
        "",
        "| Nhóm từ vựng | Số bản ghi | Ví dụ nhãn có nhiều QA dương nhất |",
        "|---|---:|---|",
    ]
    for (fam, group), g in inventory.groupby(["lexicon_family", "value_group"]):
        g = g.copy()
        g["main_positive"] = g.val_yes_support_rows.astype(int) + g.test_yes_support_rows.astype(int)
        top = g.sort_values(["main_positive", "canonical_term"], ascending=[False, True]).head(5)
        examples = "; ".join(f"{r.canonical_term} ({int(r.main_positive)})" for r in top.itertuples())
        lines.append(f"| {fam} / {group} | {len(g):,} | {examples} |")
    lines += [
        "",
        "## Phạm vi và giới hạn dữ liệu",
        "",
        "- Danh sách `Diagnosis_answer_list` lấy từ đáp án của nhóm Diagnosis; `Source_disease_label` là nhãn bệnh nguồn gắn với ảnh. Hai danh sách được giữ riêng, không coi là đồng nghĩa và không dịch hàng loạt.",
        "- Bài báo liệt kê các chiều thuộc tính như kích thước, màu, hình dạng, số lượng, phân bố và ranh giới. TSV DermNet có nhóm màu/hình dạng và trường đặc điểm rộng, nhưng không có cột riêng nhất quán cho kích thước, số lượng hoặc ranh giới; từ điển giữ theo trường nguồn thay vì tự suy ra các nhãn không được ghi rõ.",
        "- Các TSV hiện không có nhóm `Lesion_Reasoning`, `Spatial_Relation` hoặc `Suggestion & Treatment`; vì vậy không tạo danh sách cho ba năng lực đó.",
        "- Kiểm tra nhãn và cấu trúc được kế thừa từ vòng rà soát trước; không có bác sĩ thẩm định thủ công từng ảnh. Cây vị trí là bản ánh xạ ứng viên; các dòng được đánh dấu cần ánh xạ thủ công không nên dùng như quan hệ ontology đã xác nhận.",
        f"- Có {len(unmapped):,} nhãn vị trí đang để `requires_manual_body_region_mapping`. Còn {family_counts.get('Anatomical_field_other', 0):,} giá trị trong `Anatomical_Distribution` được giữ ở nhóm chưa phân loại/hỗn hợp vì có thể là vị trí, kiểu phân bố, cấu hình hoặc dữ liệu khác. Có {len(unparsed_judgements):,} câu Judgement chưa bóc tách được khái niệm mục tiêu; các câu này nằm trong `DermNet_lexical_unparsed_judgements_20260925.tsv`.",
        "",
        "## File đầu ra",
        "",
        "- `DermNet_lexical_inventory_20260925.tsv`: toàn bộ danh sách từ vựng dạng bảng, có Level 1–4, thuật ngữ, nhóm nguồn và số QA/image theo Val, Test và Test_1of3.",
        "- `DermNet_lexical_tree_20260925.json`: metadata, cây vị trí cơ thể, cây khoang miệng tách riêng và bản ghi từ vựng đầy đủ cho xử lý bằng code.",
        "- `DermNet_lexical_unparsed_judgements_20260925.tsv`: câu hỏi Có/Không chưa trích được thuật ngữ mục tiêu.",
        "",
        "## Kiểm tra phạm vi",
        "",
        f"| Split | QA | Ảnh |",
        "|---|---:|---:|",
    ]
    for split, stats in split_summary.items():
        lines.append(f"| {split} | {stats['qa_rows']:,} | {stats['images']:,} |")
    report_path = OUT / "DermNet_lexical_tree_20260925.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "outputs": [str(report_path), str(inventory_path), str(json_path), str(OUT / "DermNet_lexical_unparsed_judgements_20260925.tsv")],
        "rows": split_summary,
        "main_qa": summary["main_unique_qa_rows_val_plus_test"],
        "main_images": summary["main_unique_images_val_plus_test"],
        "families": family_counts,
        "body_regions_needing_mapping": len(unmapped),
        "judgements_unparsed": len(unparsed_judgements),
        "body_region_examples_unmapped": unmapped.canonical_term.head(30).tolist(),
        "unparsed_examples": unparsed_judgements[:3],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
