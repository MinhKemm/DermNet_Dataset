import csv
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


INVENTORY = Path(sys.argv[1])
SHAPE_AUDIT = Path(sys.argv[2])
OUT_DIR = Path(sys.argv[3])
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


inventory = read_tsv(INVENTORY)
shape_rows = read_tsv(SHAPE_AUDIT)
if len(inventory) != 3621:
    raise SystemExit(f"Expected 3621 lexical inventory rows, found {len(inventory)}")
if len(shape_rows) != 344:
    raise SystemExit(f"Expected 344 Shape audit rows, found {len(shape_rows)}")


def norm(value):
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def sentence_case(value):
    value = value.strip()
    if value and value[0].isalpha():
        return value[0].upper() + value[1:]
    return value


by_field = defaultdict(dict)
for row in inventory:
    key = (row["lexicon_family"], row["value_group"])
    by_field[key][norm(row["canonical_term"])] = row

groups = []
assigned = {}


def forms_for(family, value_group, terms):
    output = set(terms)
    for term in terms:
        row = by_field.get((family, value_group), {}).get(norm(term))
        if row:
            output.update(
                item.strip()
                for item in row.get("observed_forms", "").split("||")
                if item.strip()
            )
    return sorted(output, key=str.casefold)


def add_group(family, value_group, terms, canonical, status, note):
    present = []
    for term in terms:
        row = by_field.get((family, value_group), {}).get(norm(term))
        if row:
            present.append(row["canonical_term"])
    present = sorted(set(present), key=str.casefold)
    if len(present) < 2:
        return
    key = (family, value_group, tuple(sorted(norm(x) for x in present)))
    if any(existing["key"] == key for existing in groups):
        return
    entry = {
        "key": key,
        "family": family,
        "value_group": value_group,
        "source_terms": forms_for(family, value_group, present),
        "canonical": sentence_case(canonical),
        "status": status,
        "note": note,
    }
    groups.append(entry)
    for term in present:
        assigned[(family, value_group, norm(term))] = entry


# Reuse the focused Shape review, preserving its distinction between safe and
# provisional groups instead of inventing a second set of mappings.
shape_groups = defaultdict(list)
shape_notes = defaultdict(list)
shape_status = {}
for row in shape_rows:
    action = row["action"]
    if action not in {"Chuẩn hóa gần trùng", "Giữ làm canonical", "Ứng viên gần nghĩa, cần duyệt"}:
        continue
    group = row.get("candidate_group", "").strip()
    if not group:
        continue
    shape_groups[group].append(row["source_term"])
    shape_notes[group].append(row.get("rationale", ""))
    if action == "Ứng viên gần nghĩa, cần duyệt":
        shape_status[group] = "Cần duyệt trước"
    else:
        shape_status.setdefault(group, "Chuẩn hóa gần trùng")
for canonical, terms in sorted(shape_groups.items()):
    distinct_notes = sorted({n for n in shape_notes[canonical] if n})
    add_group(
        "Attribute_value_list",
        "Shape",
        terms,
        canonical,
        shape_status[canonical],
        " ".join(distinct_notes),
    )


# Body-region labels that differ only by the redundant generic prefix "Vùng".
body_key = ("Body_region", "Body region")
body_values = by_field[body_key]
for key, row in sorted(body_values.items()):
    term = row["canonical_term"]
    if not term.casefold().startswith("vùng "):
        continue
    base = term[5:].strip()
    base_row = body_values.get(norm(base))
    if not base_row:
        continue
    review = norm(base) == "niêm mạc"
    add_group(
        "Body_region",
        "Body region",
        [term, base_row["canonical_term"]],
        base_row["canonical_term"],
        "Cần duyệt trước" if review else "Chuẩn hóa gần trùng",
        "Bỏ tiền tố Vùng nếu câu nguồn xác nhận đây là cùng vị trí; giữ các từ chỉ mặt, bên, nếp hoặc phạm vi.",
    )


# Color compounds whose only difference is the order of the same two words.
for left, right, canonical in [
    ("Đỏ tím", "Tím đỏ", "Đỏ tím"),
    ("Nâu hồng", "Hồng nâu", "Nâu hồng"),
    ("Xám vàng", "Vàng xám", "Xám vàng"),
]:
    add_group(
        "Attribute_value_list",
        "Color",
        [left, right],
        canonical,
        "Chuẩn hóa gần trùng",
        "Hai nhãn dùng cùng hai thành tố màu theo thứ tự khác nhau; thống nhất một trật tự.",
    )


# Distribution patterns. Preserve qualifiers such as small, sparse, bilateral,
# focal, peripheral, and on a specific body surface.
distribution_groups = [
    (["Cụm", "Thành cụm", "Tụ cụm", "Tụ thành cụm", "Tập trung cụm", "Từng cụm"],
     "Thành cụm", "Cần duyệt trước",
     "Các cách diễn đạt gần nhau về tổn thương tụ thành cụm; giữ riêng Cụm nhỏ và Cụm thưa."),
    (["Đám", "Thành đám", "Tụ đám", "Tụ thành đám"],
     "Thành đám", "Cần duyệt trước",
     "Các cách diễn đạt gần nhau về tổn thương tụ thành đám."),
    (["Dạng dải", "Dải", "Thành dải", "Theo dải"],
     "Dạng dải", "Cần duyệt trước",
     "Các cách diễn đạt gần nhau; xác nhận Theo dải có cùng nghĩa với dạng dải trong câu nguồn."),
    (["Dạng tuyến", "Tuyến tính"],
     "Tuyến tính", "Cần duyệt trước",
     "Có thể cùng mô tả phân bố tuyến tính; xác nhận ngữ cảnh trước khi nhập chung."),
    (["Theo hàng", "Thành hàng", "Xếp hàng", "Xếp thành hàng"],
     "Thành hàng", "Chuẩn hóa gần trùng",
     "Cùng mô tả tổn thương xếp thành một hàng."),
    (["Nằm ngang", "Theo chiều ngang"],
     "Theo chiều ngang", "Chuẩn hóa gần trùng",
     "Cùng mô tả hướng ngang."),
    (["Thưa", "Thưa thớt"],
     "Thưa thớt", "Chuẩn hóa gần trùng",
     "Cùng mô tả mật độ thưa."),
    (["Vùng phơi nắng", "Da phơi nắng"],
     "Vùng phơi nắng", "Cần duyệt trước",
     "Gần nghĩa trong trường phân bố; xác nhận nhãn Da phơi nắng đang chỉ vùng cơ thể."),
    (["Vòng", "Thành vòng", "Xếp thành vòng"],
     "Thành vòng", "Cần duyệt trước",
     "Cùng mô tả tổn thương sắp thành vòng; giữ riêng Ngoại vi thành vòng và Đồng tâm."),
    (["Đơn độc", "Đơn ổ", "Một ổ", "Riêng lẻ"],
     "Đơn độc", "Cần duyệt trước",
     "Các nhãn gần nhau nhưng có thể khác giữa một tổn thương và một ổ bệnh."),
    (["Đa ổ", "Nhiều ổ"],
     "Nhiều ổ", "Cần duyệt trước",
     "Xác nhận cách dùng ổ và nhiều ổ theo câu nguồn."),
]
for terms, canonical, status, note in distribution_groups:
    # Terms can live in different internal subgroups of Distribution_pattern.
    present = [r for r in inventory if r["lexicon_family"] == "Distribution_pattern"
               and norm(r["canonical_term"]) in {norm(t) for t in terms}]
    subgroups = sorted({r["value_group"] for r in present})
    if len({norm(r["canonical_term"]) for r in present}) < 2:
        continue
    # The distribution inventory has one value_group per term; record it as a
    # shared field-level group so no pattern is dropped due to its subcategory.
    name = "; ".join(subgroups)
    actual_terms = sorted({r["canonical_term"] for r in present}, key=str.casefold)
    actual_forms = set(actual_terms)
    for r in present:
        actual_forms.update(x.strip() for x in r.get("observed_forms", "").split("||") if x.strip())
    groups.append({
        "key": ("Distribution_pattern", name, tuple(sorted(norm(x) for x in actual_terms))),
        "family": "Distribution_pattern",
        "value_group": name,
        "source_terms": sorted(actual_forms, key=str.casefold),
        "canonical": canonical,
        "status": status,
        "note": note,
    })
    for term in actual_terms:
        assigned[("Distribution_pattern", name, norm(term))] = groups[-1]


# Lesion terms: keep related primary lesions and qualifiers distinct. Only
# reverse-order wording and direct noun variants appear in these proposals.
lesion_groups = [
    ("Primary_Lesion_Type", ["Ban dát", "Dát ban"], "Ban dát", "Cần duyệt trước",
     "Hai cách đảo trật tự từ; xác nhận tài liệu nguồn dùng cùng một khái niệm."),
    ("Primary_Lesion_Type", ["Ban dát đỏ", "Dát ban đỏ"], "Ban dát đỏ", "Cần duyệt trước",
     "Hai cách đảo trật tự từ; xác nhận tài liệu nguồn dùng cùng một khái niệm."),
    ("Primary_Lesion_Type", ["Mụn đầu đen", "Nhân trứng cá mở"], "Nhân trứng cá mở", "Cần duyệt trước",
     "Đề xuất tên hình thái chuẩn hơn; bác sĩ xác nhận câu nguồn trước khi nhập chung."),
    ("Primary_Lesion_Type", ["Mụn đầu trắng", "Nhân trứng cá kín"], "Nhân trứng cá kín", "Cần duyệt trước",
     "Đề xuất tên hình thái chuẩn hơn; bác sĩ xác nhận câu nguồn trước khi nhập chung."),
    ("Primary_and_Secondary_Morphology", ["Trợt", "Trợt da", "Vết trợt"], "Trợt", "Cần duyệt trước",
     "Các cách gọi gần nhau; giữ riêng các nhãn có thêm độ sâu hoặc vị trí như trợt nông, trợt niêm mạc."),
    ("Primary_and_Secondary_Morphology", ["Loét", "Vết loét", "Ổ loét"], "Loét", "Cần duyệt trước",
     "Các cách gọi gần nhau; giữ riêng loét nông, loét niêm mạc và các qualifier khác."),
    ("Primary_and_Secondary_Morphology", ["Nứt", "Vết nứt"], "Nứt", "Cần duyệt trước",
     "Gần nghĩa; không nhập Nứt da, Nứt kẽ hoặc Nứt móng nếu trường vị trí có ý nghĩa."),
    ("Primary_and_Secondary_Morphology", ["Sẩn lõm giữa", "Sẩn lõm rốn"], "Sẩn lõm trung tâm", "Cần duyệt trước",
     "Có thể cùng chỉ sẩn lõm trung tâm; xác nhận thuật ngữ trong nguồn."),
]
for value_group, terms, canonical, status, note in lesion_groups:
    add_group("Lesion_list", value_group, terms, canonical, status, note)


# Other attribute groups with close word-order variants or commonly used
# synonym pairs. All uncertain clinical synonym mappings remain review-only.
attribute_groups = [
    ("Nail_Morphology", ["Biến dạng móng", "Móng biến dạng"], "Biến dạng móng", "Chuẩn hóa gần trùng",
     "Cùng cụm từ với trật tự từ khác nhau; giữ riêng Biến đổi móng."),
    ("Nail_Morphology", ["Dày móng", "Móng dày"], "Dày móng", "Chuẩn hóa gần trùng",
     "Cùng cụm từ với trật tự từ khác nhau."),
    ("Nail_Morphology", ["Loạn dưỡng móng", "Móng loạn dưỡng"], "Loạn dưỡng móng", "Chuẩn hóa gần trùng",
     "Cùng cụm từ với trật tự từ khác nhau."),
    ("Nail_Morphology", ["Rãnh dọc móng", "Rãnh móng dọc"], "Rãnh dọc móng", "Chuẩn hóa gần trùng",
     "Cùng cụm từ với trật tự từ khác nhau."),
    ("Nail_Morphology", ["Tách móng", "Ly móng"], "Tách móng", "Cần duyệt trước",
     "Đề xuất thống nhất cách gọi; bác sĩ xác nhận hai nhãn được dùng đồng nghĩa trong ngữ cảnh nguồn."),
    ("Hair_and_Surface_Characteristics", ["Tóc thưa", "Thưa tóc"], "Tóc thưa", "Chuẩn hóa gần trùng",
     "Cùng mô tả mật độ tóc thưa."),
    ("Hair_and_Surface_Characteristics", ["Lông thưa", "Thưa lông"], "Lông thưa", "Chuẩn hóa gần trùng",
     "Cùng mô tả mật độ lông thưa."),
    ("Hair_Morphology", ["Tóc gãy", "Sợi tóc gãy"], "Tóc gãy", "Cần duyệt trước",
     "Gần nghĩa; xác nhận việc bỏ từ Sợi không làm mất thông tin cần thiết."),
    ("Hair_and_Surface_Characteristics", ["Da bóng", "Bề mặt bóng"], "Bề mặt bóng", "Cần duyệt trước",
     "Gần nghĩa; giữ riêng các qualifier như bóng nhờn, bóng ẩm hoặc bóng căng."),
    ("Hair_and_Surface_Characteristics", ["Da bóng nhờn", "Bóng nhờn", "Bề mặt bóng nhờn"], "Bề mặt bóng nhờn", "Cần duyệt trước",
     "Các cách gọi gần nhau; giữ riêng Bề mặt bóng nếu không có ý nhờn."),
    ("Hair_and_Surface_Characteristics", ["Bề mặt nhẵn", "Da nhẵn", "Bề mặt trơn láng", "Da trơn láng"], "Bề mặt nhẵn", "Cần duyệt trước",
     "Gần nghĩa về bề mặt nhẵn; không gộp với Bề mặt nhăn."),
    ("Hair_and_Surface_Characteristics", ["Da khô", "Khô da"], "Da khô", "Chuẩn hóa gần trùng",
     "Cùng cụm từ với trật tự từ khác nhau."),
    ("Surface_or_secondary_change", ["Vảy rìa", "Vảy ở rìa"], "Vảy rìa", "Chuẩn hóa gần trùng",
     "Cùng mô tả vảy nằm ở rìa tổn thương."),
    ("Surface_or_secondary_change", ["Vảy mỏng", "Vảy da mỏng"], "Vảy mỏng", "Cần duyệt trước",
     "Gần nghĩa; xác nhận từ Da không mang thêm thông tin trong câu nguồn."),
    ("Surface_or_secondary_change", ["Trợt", "Trợt da", "Vết trợt"], "Trợt", "Cần duyệt trước",
     "Giữ riêng trợt nông, trợt niêm mạc và trợt loét."),
    ("Surface_or_secondary_change", ["Loét", "Vết loét", "Ổ loét"], "Loét", "Cần duyệt trước",
     "Giữ riêng loét nông, loét niêm mạc và các qualifier khác."),
    ("Surface_or_secondary_change", ["Nứt", "Vết nứt"], "Nứt", "Cần duyệt trước",
     "Giữ riêng nứt da, nứt kẽ, nứt móng và các qualifier khác."),
    ("Surface_or_secondary_change", ["Sưng nề", "Phù nề"], "Phù nề", "Cần duyệt trước",
     "Có thể là cách gọi đồng nghĩa; giữ riêng sưng theo vị trí hoặc mức độ."),
    ("Surface_or_secondary_change", ["Phù mi", "Phù mi mắt"], "Phù mi mắt", "Chuẩn hóa gần trùng",
     "Cùng vị trí giải phẫu; dùng dạng đầy đủ Phù mi mắt."),
    ("Surface_or_secondary_change", ["Da macerê", "Macer hóa", "Mềm hóa do ẩm"], "Mềm hóa do ẩm", "Cần duyệt trước",
     "Đề xuất một cách gọi tiếng Việt; giữ các qualifier mức độ như nhẹ hoặc ít."),
]
for value_group, terms, canonical, status, note in attribute_groups:
    add_group("Attribute_value_list", value_group, terms, canonical, status, note)


# Include every casing-only source-form cluster not already represented above.
case_clusters = []
for row in inventory:
    family, value_group = row["lexicon_family"], row["value_group"]
    raw_forms = list(dict.fromkeys(
        item.strip() for item in row.get("observed_forms", "").split("||") if item.strip()
    ))
    if len(raw_forms) < 2:
        continue
    if len({norm(item) for item in raw_forms}) != 1:
        continue
    term_key = (family, value_group, norm(row["canonical_term"]))
    existing = assigned.get(term_key)
    canonical = row["canonical_term"].strip()
    if canonical and canonical[0].islower():
        canonical = canonical[0].upper() + canonical[1:]
    if existing:
        existing["source_terms"] = sorted(set(existing["source_terms"] + raw_forms), key=str.casefold)
        continue
    case_clusters.append({
        "family": family,
        "value_group": value_group,
        "source_terms": sorted(raw_forms, key=str.casefold),
        "canonical": canonical,
        "status": "Chỉ chuẩn hóa chữ hoa/thường",
        "note": "Giữ nguyên từ và qualifier; thống nhất chữ cái đầu viết hoa.",
    })


# Add case-only variants to the auditable mapping table; retain the mapping
# proposal as one row per lexical field, not one row per individual spelling.
all_groups = groups + case_clusters
all_groups.sort(key=lambda g: (g["family"], g["value_group"], g["canonical"].casefold()))

FAMILY_LABELS = {
    "Anatomical_field_other": "Vị trí/giá trị lẫn trường",
    "Attribute_value_list": "Thuộc tính tổn thương",
    "Body_region": "Vị trí cơ thể",
    "Diagnosis_answer_list": "Bệnh danh trong đáp án",
    "Distribution_pattern": "Kiểu phân bố",
    "Lesion_list": "Loại tổn thương",
    "Oral_region": "Vị trí khoang miệng",
    "Source_disease_label": "Bệnh danh nguồn",
}
VALUE_LABELS = {
    "Unmapped_or_mixed_value_from_Anatomical_Distribution": "Giá trị chưa phân loại trong trường phân bố",
    "Body region": "Vùng cơ thể",
    "Diagnosis": "Chẩn đoán",
    "Source disease": "Bệnh nguồn",
    "Color": "Màu sắc",
    "Shape": "Hình dạng",
    "Hair_and_Surface_Characteristics": "Tóc và bề mặt da",
    "Hair_Morphology": "Hình thái tóc/lông",
    "Nail_Morphology": "Hình thái móng",
    "Secondary_Change": "Biến đổi thứ phát",
    "Surface_or_secondary_change": "Bề mặt/biến đổi thứ phát",
    "Swelling": "Sưng/phù",
    "Primary_Lesion_Type": "Tổn thương cơ bản",
    "Primary_and_Secondary_Morphology": "Tổn thương cơ bản và thứ phát",
    "Oral cavity": "Khoang miệng/niêm mạc",
    "Annular_or_circular_arrangement": "Phân bố dạng vòng",
    "Body_surface_orientation": "Mặt cơ thể",
    "Clustered": "Tụ cụm/đám",
    "Density": "Mật độ",
    "Exposure_or_contact": "Phơi nắng/tiếp xúc",
    "Extent": "Phạm vi",
    "Extent_and_configuration": "Phạm vi/cấu hình",
    "Extent_and_laterality": "Phạm vi/bên",
    "Flexural_or_extensor": "Mặt gấp/duỗi",
    "Flexural_or_linear": "Nếp gấp/dạng tuyến",
    "Follicular_pattern": "Kiểu nang lông",
    "Irregular_or_asymmetric_pattern": "Không đều/bất đối xứng",
    "Laterality_and_symmetry": "Bên/đối xứng",
    "Linear_or_row": "Dạng tuyến/dải/hàng",
    "Number": "Số lượng",
    "Periungual_pattern": "Quanh móng",
    "Reticular_or_network_pattern": "Dạng lưới",
    "Spatial_or_configuration_candidate": "Vị trí/cấu hình",
}


def family_label(value):
    return FAMILY_LABELS.get(value, value)


def value_label(value):
    parts = [VALUE_LABELS.get(part.strip(), part.strip()) for part in value.split("; ")]
    return "; ".join(parts)

tsv_path = OUT_DIR / "DermNet_All_fields_normalization_audit_20260925.tsv"
with tsv_path.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["nhom_truong", "nhom_con", "tu_dang_dung", "tu_chuan_de_xuat", "trang_thai", "ghi_chu", "lexicon_family_key", "value_group_key"],
        delimiter="\t",
    )
    writer.writeheader()
    for group in all_groups:
        writer.writerow({
            "nhom_truong": family_label(group["family"]),
            "nhom_con": value_label(group["value_group"]),
            "tu_dang_dung": " || ".join(group["source_terms"]),
            "tu_chuan_de_xuat": group["canonical"],
            "trang_thai": group["status"],
            "ghi_chu": group["note"],
            "lexicon_family_key": group["family"],
            "value_group_key": group["value_group"],
        })


family_counts = Counter(row["lexicon_family"] for row in inventory)
case_counts = Counter(group["family"] for group in case_clusters)
semantic_counts = Counter(group["family"] for group in groups)
all_fields = sorted(family_counts)
table_rows = []
for family in all_fields:
    table_rows.append(
        f"| {family} | {family_counts[family]} | {semantic_counts[family]} | {case_counts[family]} |"
    )

def md_table(headers, data):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in data)
    return "\n".join(lines)


proposal_rows = [
    [family_label(g["family"]), value_label(g["value_group"]), " / ".join(g["source_terms"]), g["canonical"], g["status"], g["note"]]
    for g in groups
]
case_examples = []
for family in all_fields:
    examples = [g for g in case_clusters if g["family"] == family]
    sample = examples[:5]
    example_text = "; ".join(
        f"{' / '.join(g['source_terms'])} → {g['canonical']}" for g in sample
    )
    case_examples.append([family_label(family), len(examples), example_text or "Không có biến thể chữ hoa/thường được ghi nhận."])

report = [
    "# Rà soát từ gần trùng trên toàn bộ lexical inventory DermNet",
    "",
    "## Phạm vi",
    "",
    f"Rà toàn bộ {len(inventory):,} bản ghi lexical thuộc {len(all_fields)} nhóm trường, không chỉ Shape. Inventory được dựng từ Val_4k và Test; Test 1/3 là tập con của Test nên không cộng thành tập nguồn thứ ba.",
    "",
    "Báo cáo chỉ đề xuất gộp các cách viết rất gần hoặc biến thể chữ hoa/thường. Các nhóm có thể khác nghĩa lâm sàng được đánh dấu Cần duyệt trước. File nguồn QA không bị sửa.",
    "",
    "## Bao phủ rà soát",
    "",
    md_table(["Nhóm trường", "Bản ghi inventory", "Nhóm biến thể/gần nghĩa", "Cụm chỉ khác hoa/thường"],
             [[family_label(family), family_counts[family], semantic_counts[family], case_counts[family]] for family in all_fields]),
    "",
    "## Các nhóm biến thể và từ chuẩn đề xuất",
    "",
    md_table(["Nhóm trường", "Nhóm con", "Các từ đang dùng", "Từ chuẩn đề xuất", "Xử lý", "Ghi chú"], proposal_rows),
    "",
    "## Khác biệt chỉ ở chữ hoa/thường",
    "",
    "Các nhóm này giữ nguyên từ và qualifier; chỉ thống nhất chữ cái đầu viết hoa. TSV kèm theo liệt kê từng cụm biến thể.",
    "",
    md_table(["Nhóm trường", "Số cụm", "Ví dụ"], case_examples),
    "",
    "## Không gộp tự động",
    "",
    "- Bệnh danh gần giống chữ vẫn được giữ riêng nếu chỉ khác nhau ở vị trí, thể bệnh hoặc mức độ; cần đối chiếu nhãn nguồn trước khi chuẩn hóa.",
    "- Ví dụ, không gộp Nodular melanoma với Ocular melanoma chỉ vì cùng chứa melanoma; đây là các nhãn chẩn đoán khác nhau.",
    "- Giữ riêng các cặp hình thái có ranh giới chuyên môn, như dát/mảng, sẩn/nốt, mụn nước/bọng nước, trợt/loét; không dùng độ giống từ để nhập chung.",
    "- Giữ nguyên các qualifier như nhỏ, lớn, nông, sâu, nhẹ, nhạt, sẫm, bên, quanh, ngoại vi, trung tâm, nhiều và ít.",
    "- Anatomical_field_other là nhóm giá trị hỗn hợp/chưa phân loại; chỉ chuẩn hóa viết hoa/thường, không diễn giải thành vị trí cơ thể.",
    "- DermNet phân biệt các hình thái này trong mục [Terminology in dermatology](https://dermnetnz.org/topics/terminology).",
    "",
    "TSV có toàn bộ nhóm mapping được đề xuất, gồm biến thể chữ hoa/thường và các nhóm gần nghĩa. Các dòng Cần duyệt trước là ứng viên để người rà soát quyết định; chưa áp dụng vào các file QA.",
]
md_path = OUT_DIR / "DermNet_All_fields_normalization_audit_20260925.md"
md_path.write_text("\n".join(report) + "\n", encoding="utf-8")

print(f"Inventory rows: {len(inventory)}; lexical families: {len(all_fields)}")
print(f"Candidate groups: {len(groups)}; case-only clusters: {len(case_clusters)}; total report rows: {len(all_groups)}")
for family in all_fields:
    print(f"{family}: inventory={family_counts[family]}, near_groups={semantic_counts[family]}, case_only={case_counts[family]}")
print(f"Wrote: {md_path}")
print(f"Wrote: {tsv_path}")
