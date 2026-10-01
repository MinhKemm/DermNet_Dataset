import csv
import sys
from collections import defaultdict
from pathlib import Path


SOURCE = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def read_source():
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows = [
            row for row in csv.DictReader(f, delimiter="\t")
            if row["lexicon_family"] == "Attribute_value_list"
            and row["value_group"] == "Shape"
        ]
    if len(rows) != 344:
        raise SystemExit(f"Expected 344 Shape terms, found {len(rows)}")
    return rows


def integer(row, key):
    try:
        return int(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


# Only wording variants and very close synonyms are eligible for normalization.
# Qualifiers and labels with potentially different meaning remain separate.
AUTO_GROUPS = [
    ("Hình bầu dục", ["Bầu dục", "Dạng bầu dục", "Hình bầu dục"],
     "Cùng một hình dạng; khác tiền tố Hình/Dạng."),
    ("Hình tròn", ["tròn", "Hình tròn"],
     "Cùng một hình dạng; khác tiền tố Hình và chữ hoa/thường."),
    ("Hình chữ nhật", ["Chữ nhật", "hình chữ nhật"],
     "Cùng một hình dạng; khác tiền tố Hình và chữ hoa/thường."),
    ("Hình tam giác", ["Tam giác", "Hình tam giác"],
     "Cùng một hình dạng; khác tiền tố Hình."),
    ("Hình thoi", ["Thoi", "Hình thoi"],
     "Cùng một hình dạng; khác tiền tố Hình."),
    ("Hình đa giác", ["Đa giác", "Hình đa giác"],
     "Cùng một hình dạng; khác tiền tố Hình."),
    ("Dạng vòm", ["Vòm", "Dạng vòm", "hình vòm"],
     "Cùng khái niệm vòm; khác tiền tố hoặc chữ hoa/thường."),
    ("Hình bán cầu", ["Bán cầu", "Hình bán cầu"],
     "Cùng cụm từ; khác tiền tố Hình."),
    ("Hình cung", ["Cung", "Hình cung", "Dạng cung"],
     "Cùng mô tả hình cung; khác tiền tố."),
    ("Hình vòng cung", ["Vòng cung", "Hình vòng cung"],
     "Cùng cụm từ; khác tiền tố Hình."),
    ("Dạng bản đồ", ["Bản đồ", "Hình bản đồ", "Dạng bản đồ"],
     "Cùng mô tả dạng bản đồ; khác tiền tố."),
    ("Dạng dải", ["Dải", "Dạng dải"],
     "Cùng cụm từ; khác tiền tố Dạng."),
    ("Dạng đường", ["Đường", "Dạng đường"],
     "Cùng cụm từ; khác tiền tố Dạng."),
    ("Dạng đường thẳng", ["Đường thẳng", "Dạng đường thẳng"],
     "Cùng cụm từ; khác tiền tố Dạng."),
    ("Dạng vòng", ["vòng", "Dạng vòng", "Hình vòng"],
     "Cùng cách gọi hình vòng; giữ riêng nhẫn và các qualifier vòng hở/một phần."),
    ("Dạng mạng lưới", ["lưới", "Mạng lưới", "Dạng lưới", "dạng mạng lưới"],
     "Cùng cách gọi dạng lưới/mạng lưới; giữ riêng vân lưới mờ và dạng ô lưới."),
    ("Hình đồng xu", ["Đồng xu", "Đồng tiền", "Hình đồng xu"],
     "Cùng cách gọi hình đồng xu/đồng tiền; không gộp với hình tròn chung."),
    ("Hình cánh bướm", ["Cánh bướm", "Hình cánh bướm"],
     "Cùng cụm từ; khác tiền tố Hình."),
    ("Hình tròn nhỏ", ["Tròn nhỏ", "Hình tròn nhỏ"],
     "Cùng hình dạng và qualifier nhỏ; không gộp với Hình tròn."),
    ("Bầu dục kéo dài", ["Bầu dục dài", "Bầu dục kéo dài"],
     "Cùng hình dạng và qualifier kéo dài; không gộp với Bầu dục."),
    ("Lõm trung tâm", ["Lõm giữa", "Trung tâm lõm", "Lõm trung tâm"],
     "Cùng mô tả vùng lõm ở trung tâm; khác trật tự từ."),
    ("Ranh giới không rõ", ["Ranh giới mờ", "Ranh giới không rõ", "Bờ mờ",
                            "Bờ không rõ", "Mờ ranh giới"],
     "Cùng mô tả ranh giới không rõ; khác cách diễn đạt."),
    ("Ranh giới rõ", ["Ranh giới rõ", "bờ rõ", "Bờ viền rõ"],
     "Cùng mô tả ranh giới rõ; khác cách diễn đạt."),
    ("Ranh giới tương đối rõ", ["Bờ khá rõ", "Bờ tương đối rõ",
                                "Giới hạn khá rõ", "Ranh giới tương đối rõ"],
     "Các cách diễn đạt gần nhau về ranh giới tương đối rõ."),
    ("Ranh giới không đều", ["Ranh giới không đều", "Bờ không đều"],
     "Cùng mô tả bờ/ranh giới không đều."),
    ("Không đối xứng", ["Bất đối xứng", "Không đối xứng"],
     "Cùng nghĩa; khác tiền tố phủ định."),
    ("Bờ nham nhở", ["Bờ nham nhở", "Mép nham nhở", "Rìa nham nhở"],
     "Cùng mô tả bờ nham nhở; giữ riêng từ Nham nhở nếu thiếu ngữ cảnh."),
    ("Vòng không hoàn toàn", ["Vòng không hoàn toàn", "Vòng một phần", "Vòng hở"],
     "Cùng mô tả vòng không khép kín; giữ nguyên qualifier không hoàn toàn."),
    ("Tròn bầu dục", ["Tròn bầu dục", "Tròn-bầu dục"],
     "Chỉ chuẩn hóa dấu nối; không gộp với Bầu dục."),
]

REVIEW_GROUPS = [
    ("Không đều", ["bất quy tắc", "Không đều"],
     "Có thể gần nghĩa nhưng cần xác nhận source context trước khi gộp."),
    ("Bia đích", ["Bia bắn", "Bia đích", "Hình bia bắn", "Dạng bia"],
     "Các nhãn gần nhau nhưng mức độ tương đương chưa chắc chắn."),
    ("Dạng vòng", ["Nhẫn", "Hình nhẫn", "Dạng nhẫn"],
     "Gần nghĩa với Dạng vòng; cần xác nhận có dùng thay thế cho nhau trong bộ dữ liệu."),
    ("Dạng thùy múi", ["Dạng thùy", "Dạng múi", "Thùy múi", "Nhiều thùy",
                       "đa thùy", "Thuỳ"],
     "Các cụm gần nghĩa nhưng có thể khác nhau về số lượng/mức độ thùy."),
    ("Dạng chùm", ["Chùm", "Dạng chùm", "Thành chùm"],
     "Các cách nói gần nhau; kiểm tra câu nguồn trước khi thống nhất cách diễn đạt."),
    ("Bờ nham nhở", ["Nham nhở"],
     "Có thể là bờ nham nhở nhưng nhãn thiếu từ chỉ vị trí."),
]


def make_map(groups, status, confidence):
    result = {}
    for canonical, terms, rationale in groups:
        for term in terms:
            result[term.casefold()] = {
                "canonical": canonical,
                "status": status,
                "confidence": confidence,
                "rationale": rationale,
            }
    return result


AUTO = make_map(AUTO_GROUPS, "Chuẩn hóa gần trùng", "Cao")
REVIEW = make_map(REVIEW_GROUPS, "Ứng viên gần nghĩa, cần duyệt", "Trung bình")


def build_rows(source):
    result = []
    for row in source:
        term = row["canonical_term"]
        key = term.casefold()
        proposal = AUTO.get(key) or REVIEW.get(key)
        if proposal:
            canonical = proposal["canonical"]
            if proposal["status"] == "Chuẩn hóa gần trùng" and key == canonical.casefold():
                action = "Giữ làm canonical"
                rationale = "Giá trị canonical của nhóm; không cần sửa nhãn này."
            else:
                action = proposal["status"]
                rationale = proposal["rationale"]
            confidence = proposal["confidence"]
            group = canonical
        else:
            canonical = term
            action = "Giữ nguyên"
            confidence = ""
            group = ""
            rationale = "Không đủ giống để gộp an toàn; giữ nguyên cả nghĩa và qualifier."
        result.append({
            "source_term": term,
            "observed_forms": row.get("observed_forms", ""),
            "action": action,
            "candidate_canonical_term": canonical,
            "confidence": confidence,
            "candidate_group": group,
            "gold_rows_val": integer(row, "val_gold_answer_rows"),
            "gold_rows_test": integer(row, "test_gold_answer_rows"),
            "gold_rows_test_1of3": integer(row, "test_1of3_gold_answer_rows"),
            "mcq_option_rows_val": integer(row, "val_mcq_option_rows"),
            "mcq_option_rows_test": integer(row, "test_mcq_option_rows"),
            "mcq_option_rows_test_1of3": integer(row, "test_1of3_mcq_option_rows"),
            "images_val": integer(row, "val_images"),
            "images_test": integer(row, "test_images"),
            "images_test_1of3": integer(row, "test_1of3_images"),
            "rationale": rationale,
            "source_mapping_status": row.get("mapping_status", ""),
        })
    return result


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(value).replace("|", "/").replace("\n", " ") for value in row
        ) + " |")
    return "\n".join(lines)


def write_report(rows, out_path):
    counts = defaultdict(int)
    for row in rows:
        counts[row["action"]] += 1
    normalized = [r for r in rows if r["action"] == "Chuẩn hóa gần trùng"]
    canonical_rows = [r for r in rows if r["action"] == "Giữ làm canonical"]
    review = [r for r in rows if r["action"] == "Ứng viên gần nghĩa, cần duyệt"]
    groups = defaultdict(list)
    for row in normalized + canonical_rows:
        groups[row["candidate_group"]].append(row["source_term"])
    review_groups = defaultdict(list)
    for row in review:
        review_groups[row["candidate_group"]].append(row["source_term"])
    all_groups = defaultdict(lambda: {"terms": [], "review": [], "note": ""})
    for row in normalized + canonical_rows:
        item = all_groups[row["candidate_group"]]
        item["terms"].append(row["source_term"])
        if row["action"] == "Chuẩn hóa gần trùng":
            item["note"] = row["rationale"]
        elif not item["note"]:
            item["note"] = row["rationale"]
    for row in review:
        item = all_groups[row["candidate_group"]]
        item["terms"].append(row["source_term"])
        item["review"].append(row["source_term"])
        if not item["note"]:
            item["note"] = row["rationale"]
    group_table_rows = []
    for canonical, item in sorted(all_groups.items()):
        status = "Chuẩn hóa các biến thể gần trùng"
        note = item["note"]
        if item["review"]:
            status = "Duyệt các từ này trước khi gộp"
            review_terms = ", ".join(sorted(set(item["review"]), key=str.casefold))
            note = f"{note} Cần xác nhận: {review_terms}."
        group_table_rows.append([
            ", ".join(sorted(set(item["terms"]), key=str.casefold)),
            canonical,
            status,
            note,
        ])

    report = [
        "# Rà soát chuẩn hóa gần trùng trong Shape",
        "",
        "## Kết luận",
        "",
        f"Đã rà đủ {len(rows)} nhãn Shape. Chỉ đề xuất chuẩn hóa khi khác biệt chủ yếu là cách viết, tiền tố Hình/Dạng, trật tự từ hoặc một biến thể từ ngữ rất gần. Các nhãn có qualifier hoặc có thể mang nghĩa khác được giữ nguyên hoặc đánh dấu để duyệt.",
        "",
        "Báo cáo này không đề xuất chuyển nhãn sang trường khác, không thay đổi cấu trúc dữ liệu và không chỉnh sửa câu hỏi/đáp án nguồn.",
        "",
        "## Tóm tắt",
        "",
        f"- Tổng nhãn được rà: **{len(rows)}**.",
        f"- Nhãn cần đổi sang canonical độ tin cậy cao: **{len(normalized)}**.",
        f"- Nhãn được giữ làm canonical của các nhóm đó: **{len(canonical_rows)}**.",
        f"- Nhãn thuộc nhóm ứng viên gần nghĩa cần duyệt: **{len(review)}**.",
        f"- Nhãn giữ nguyên: **{counts['Giữ nguyên']}**.",
        f"- Nhóm chuẩn hóa gần trùng: **{len(groups)}**.",
        f"- Nhóm ứng viên cần duyệt: **{len(review_groups)}**.",
        "",
        "TSV có số lần xuất hiện riêng cho Val, Test và Test 1/3. Không cộng các cột thành số ảnh duy nhất vì các tập có thể chồng lặp.",
        "",
        "## Các nhóm từ gần nhau và canonical đề xuất",
        "",
        table(
            ["Các từ đang dùng", "Từ chuẩn đề xuất", "Xử lý", "Ghi chú"],
            group_table_rows,
        ),
        "",
        "Các nhóm ghi Duyệt các từ này trước khi gộp là ứng viên, chưa áp dụng tự động. Bất quy tắc và Không đều chỉ nên nhập chung sau khi xác nhận câu nguồn; Tròn-bầu dục chỉ đổi dấu nối thành Tròn bầu dục, không nhập với Bầu dục.",
        "",
        "## Giữ nguyên các khác biệt có ý nghĩa",
        "",
        "- Giữ qualifier như nhỏ, kéo dài, một phần, hở, rõ, mờ, nhẹ, thấp và ngoằn ngoèo.",
        "- Không gộp hình tròn với hình đồng xu, hình cầu với hình tròn, vòng hoàn chỉnh với vòng hở, hoặc hình bầu dục với bầu dục kéo dài.",
        "- Không đổi nhãn trong dữ liệu nguồn cho đến khi nhóm duyệt mapping.",
        "",
        "## Cách dùng bảng",
        "",
        "Lọc cột action theo Chuẩn hóa gần trùng để xem các ánh xạ có thể áp dụng trên bản sao. Lọc theo Ứng viên gần nghĩa, cần duyệt để xác nhận bằng source context. Các dòng Giữ nguyên không cần chuẩn hóa theo đợt này.",
        "",
        "## Kiểm tra",
        "",
        "Bảng giữ đủ một dòng cho mỗi nhãn nguồn và bảo toàn số đếm tách theo từng split. Đây là rà soát từ vựng và mức độ gần nhau của nhãn, không phải thẩm định dấu hiệu trên từng ảnh.",
    ]
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return counts, groups, review_groups


def main():
    source = read_source()
    rows = build_rows(source)
    OUT.mkdir(parents=True, exist_ok=True)
    tsv_path = OUT / "DermNet_Shape_normalization_audit_20260925.tsv"
    with tsv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    md_path = OUT / "DermNet_Shape_normalization_audit_20260925.md"
    counts, groups, review_groups = write_report(rows, md_path)
    if len(rows) != 344 or len({row["source_term"] for row in rows}) != 344:
        raise SystemExit("Validation failed: audit must contain 344 unique source terms.")
    if sum(counts.values()) != 344:
        raise SystemExit("Validation failed: action counts do not reconcile.")
    print("rows =", len(rows))
    print("actions =", dict(counts))
    print("automatic_groups =", len(groups))
    print("review_groups =", len(review_groups))
    print("tsv =", tsv_path)
    print("md =", md_path)


if __name__ == "__main__":
    main()
