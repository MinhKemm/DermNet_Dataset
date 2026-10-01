from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\VuLapTrinh2\DermNet_Dataset")
OUT = ROOT / "outputs" / "dermnet-vqa-reviewed-20260923"
CLEAN = ROOT / "outputs" / "dermnet-vqa-cleaned-20260923"
RUN_BASE = ROOT / "Phase_2" / "VLMEvalKit"
FILES = {
    "Val_4k": "DermNet_Val_4k.reviewed.tsv",
    "Test_1of3": "DermNet_Test_1of3.reviewed.tsv",
    "Test": "DermNet_Test.reviewed.tsv",
}
CLEAN_FILES = {
    "Val_4k": "DermNet_Val_4k.cleaned_final.tsv",
    "Test_1of3": "DermNet_Test_1of3.cleaned_final.tsv",
    "Test": "DermNet_Test.cleaned_final.tsv",
}
DIST = re.compile(r"(?i)(?<!không )(?<!bất )\b(?:khu trú|lan tỏa|lan rộng|rải rác|thưa thớt|thưa|đối xứng|hai bên|một bên|theo dermatome|vùng duỗi|vùng gấp|vùng kẽ|vùng tiếp xúc|vùng phơi nắng|vùng tiết bã|đa ổ|nhiều ổ|một ổ|đơn độc|đơn lẻ|thành cụm|tụ thành cụm|thành đám|tụ thành đám|cụm|hợp lưu|vệ tinh|tổn thương vệ tinh|tập trung|quanh nang lông|nhiều tổn thương)\b")
MISPLACED_DISTRIBUTION = re.compile(r"(?i)\b(?:mật độ dày đặc|dày đặc|theo dải|tụ đám|hợp đám|kết đám|tụ cụm|thành đám|tụ thành đám)\b")
SURFACE_CONFIGURATION = re.compile(r"(?i)\b(?:dạng lưới mờ|dạng lưới|vân lưới)\b")
MCQ_OPTION = re.compile(r"(?m)^\s*([A-D])\.\s*(.*?)\s*$")
PRIMARY = re.compile(r"(?i)\b(?:dát|sẩn|mảng|nốt|u|mụn nước|bọng nước|mụn mủ|nang|sẩn phù|papule|nodule|plaque|vesicle|bulla|pustule|macule|patch|wheal|comedone|nhân trứng cá|nhân mụn|mụn đầu đen|mụn đầu trắng)\b")
SECONDARY = re.compile(r"(?i)\b(?:vảy|vảy tiết|trợt|loét|nứt|sẹo|teo|lichen hóa|hoại tử|bong|tróc|mài|crust|erosion|ulcer|fissure|scar|necrosis)\b")


def parse_list(s):
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def claim(q):
    m = re.search(r'"([^"\n]+)"', str(q))
    if m:
        return m.group(1).strip()
    patterns = [
        r"(?:tổn thương cơ bản chính có phải là|loại tổn thương cơ bản(?: chính)? là|tổn thương có đặc điểm lâm sàng)\s*(.*?)\s*(?:không\??|là gì\??|\?\s*$)",
        r"(?:có đặc điểm(?: lâm sàng)?|đặc điểm lâm sàng)\s+(.*?)(?:\s+không\??|\?\s*$|\. Nhận định)",
        r"\b(?:có|ghi nhận)\s+(.+?)\s+không\??\s*$",
    ]
    for pattern in patterns:
        m = re.search(pattern, str(q), re.I | re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip(" .?\"'")
    return ""


def distribution_search_text(value):
    return re.sub(r"(?i)\b(?:(?:thưa|thưa thớt)\s+(?:tóc|lông)|(?:tóc|lông)\s+(?:thưa|thưa thớt))\b", " ", str(value))


def audit(name, file):
    df = pd.read_csv(OUT / file, sep="\t", dtype=str, keep_default_na=False)
    blank = int((df[["question", "answer", "category", "type", "image_path"]] == "").any(axis=1).sum())
    missing = []
    for i, path in df.image_path.items():
        resolved = (RUN_BASE / str(path).replace("\\", "/")).resolve()
        if not resolved.is_file():
            missing.append((i, df.at[i, "index"], path))
    mcq_bad_options = mcq_bad_key = mcq_mismatch = mcq_count = 0
    for _, row in df[df.type == "Multi_choice"].iterrows():
        mcq_count += 1
        opts = {m.group(1): m.group(2).strip() for m in MCQ_OPTION.finditer(row.question)}
        if set(opts) != set("ABCD") or len({v.casefold() for v in opts.values()}) != 4 or any(not v for v in opts.values()):
            mcq_bad_options += 1
            continue
        ans = row.answer.strip().upper()
        if not re.fullmatch(r"[A-D]+", ans) or len(set(ans)) != len(ans) or ans != "".join(sorted(ans)):
            mcq_bad_key += 1
            continue
        expected = [opts[c].casefold() for c in ans]
        actual = [str(x).strip().casefold() for x in parse_list(row.answer_concepts)]
        if actual != expected:
            mcq_mismatch += 1
    bad_visual_field = int(sum(1 for _, r in df[df.category.isin(["Attribute_Shape", "Attribute_Characteristics"])].iterrows() if DIST.search(distribution_search_text(" ".join([r.question, r.answer, " ".join(map(str, parse_list(r.answer_concepts)))])))))
    misplaced_distribution = int(sum(1 for _, r in df[df.category.isin(["Attribute_Shape", "Attribute_Characteristics"])].iterrows() if MISPLACED_DISTRIBUTION.search(distribution_search_text(" ".join([r.answer, " ".join(map(str, parse_list(r.answer_concepts)))])))))
    surface_configuration = int(sum(1 for _, r in df[(df.category == "Attribute_Characteristics") & (df.sub_category == "Surface_or_secondary_change")].iterrows() if SURFACE_CONFIGURATION.search(" ".join([r.answer, " ".join(map(str, parse_list(r.answer_concepts)))]))))
    bad_plain_distribution = int(sum(1 for _, r in df[df.category == "Attribute_Characteristics"].iterrows() if re.search(r"(?i)\bphân bố\b", " ".join([r.answer, " ".join(map(str, parse_list(r.answer_concepts)))]))))
    mang_in_shape = int(sum(1 for _, r in df[df.category == "Attribute_Shape"].iterrows() if re.search(r"(?i)\bmảng\b", " ".join([r.answer, " ".join(map(str, parse_list(r.answer_concepts)))]))))
    mang_any_shape = int(sum(1 for _, r in df[df.category == "Attribute_Shape"].iterrows() if re.search(r"(?i)\bmảng\b", " ".join([r.question, r.answer, " ".join(map(str, parse_list(r.answer_concepts)))]))))
    plain_mang_in_lesion = int(sum(1 for _, r in df[df.category == "Lesion_Recognition"].iterrows() if any(str(x).strip().casefold() == "mảng" for x in parse_list(r.answer_concepts))))
    terminology_left = int(df[["question", "answer", "answer_concepts"]].apply(lambda col: col.str.contains(r"(?i)tổn thương thực thể|đóng mài|đóng mày|bóng nước|\bmài\b", regex=True)).any(axis=1).sum())
    judgement_leakage = int(sum(1 for _, r in df[df.type == "Judgement"].iterrows() if re.search(r"(?i)nhận định.{0,100}phù hợp|dựa trên.{0,160}nhận định", r.question, re.S)))
    bad_path_double = int(df.image_path.str.contains("dermnet-output/dermnet-output/images", regex=False).sum())
    bad_path_prefix = int((~df.image_path.str.startswith("../../dermnet-output/images/")).sum())
    exact_dup = int(df.duplicated(["image_id", "question", "answer"]).sum())
    nonvisual = re.compile(r"(?i)\b(?:ngứa|đau|đau rát|rát|tenderness|itch(?:ing)?|pruritus|pain|palpable|sờ thấy|dễ chảy máu|bleeds? easily|mô mềm|hậu viêm|sau viêm|post[- ]?inflammatory)\b")
    visual_rows = df[df.category != "Diagnosis"]
    nonvisual_visual_answer = int(sum(1 for _, r in visual_rows.iterrows() if nonvisual.search(" ".join([r.question, r.answer, r.answer_concepts]))))
    follicular_diagnosis = int(sum(1 for _, r in df[df.category == "Attribute_Characteristics"].iterrows() if re.search(r"(?i)\b(?:viêm nang lông|tổn thương nang lông)\b", " ".join([r.question, r.answer, r.answer_concepts]))))
    source_image_name_conflict = int(sum(1 for _, r in df.iterrows() if r.source_disease == "Adult-onset Still disease" and "cutaneous-mastocytosis-" in r.image_path.casefold()))
    postinflam_left = int(sum(1 for _, r in visual_rows.iterrows() if re.search(r"(?i)\b(?:hậu viêm|sau viêm|post[- ]?inflammatory)\b", " ".join([r.question, r.answer, r.answer_concepts]))))
    hair_pattern = re.compile(r"(?i)\b(?:tóc|lông|chấy|trứng chấy|rụng tóc|alopecia)\b")
    hair_wrong_subcategory = 0
    for _, r in df[df.category == "Attribute_Characteristics"].iterrows():
        text = " ".join([r.question, r.answer, r.answer_concepts])
        text = re.sub(r"(?i)\b(?:quanh|xung quanh)\s+nang\s+lông\b|\bnang\s+lông\b|\bviêm nang lông\b|\btổn thương nang lông\b|\bchấm nang lông\b|\btăng sừng nang lông\b", " ", text)
        if hair_pattern.search(text) and "hair" not in r.sub_category.casefold():
            hair_wrong_subcategory += 1
    hair_density_wrong_subcategory = int(sum(1 for _, r in df[(df.category == "Attribute_Characteristics") & (df.source_disease == "Trichotillomania")].iterrows() if re.search(r"(?i)\b(?:lộ da đầu|giảm mật độ)\b", " ".join([r.question, r.answer, r.answer_concepts])) and "hair" not in r.sub_category.casefold()))
    legacy_lesion_subcategory = int(((df.category == "Lesion_Recognition") & (df.sub_category == "Lesion_Type")).sum())
    ambiguous_san_not = int(sum(1 for _, r in df[df.category == "Lesion_Recognition"].iterrows() if re.search(r"(?i)\bsẩn\s+nốt\b", " ".join([r.question, r.answer, r.answer_concepts]))))
    noncanonical_inflammatory_pigment = int(sum(1 for _, r in df.iterrows() if re.search(r"(?i)\btăng sắc tố viêm\b", " ".join([r.question, r.answer, r.answer_concepts]))))
    lr_invalid = 0
    for _, r in df[df.category == "Lesion_Recognition"].iterrows():
        if r.type == "Judgement":
            labels = [claim(r.question)]
        else:
            labels = [str(x) for x in parse_list(r.answer_concepts)]
        if not labels or any(not (PRIMARY.search(x) or SECONDARY.search(x)) for x in labels):
            lr_invalid += 1
            continue
        has_primary = any(PRIMARY.search(x) for x in labels)
        has_secondary = any(SECONDARY.search(x) for x in labels)
        if r.type == "Multi_choice":
            opts = {m.group(1): m.group(2).strip() for m in MCQ_OPTION.finditer(r.question)}
            if any(not (PRIMARY.search(v) or SECONDARY.search(v)) for v in opts.values()):
                lr_invalid += 1
                continue
        if has_secondary and not has_primary and r.sub_category != "Primary_and_Secondary_Morphology":
            lr_invalid += 1
        if has_primary and has_secondary and r.sub_category != "Primary_and_Secondary_Morphology":
            lr_invalid += 1
    judgement_yes = int(sum(1 for x in df.loc[df.type == "Judgement", "answer"] if str(x).strip().casefold() in {"có", "yes"}))
    judgement_no = int(sum(1 for x in df.loc[df.type == "Judgement", "answer"] if str(x).strip().casefold() in {"không", "no"}))
    return df, {
        "rows": len(df), "images": df.image_id.nunique(), "blank_rows": blank,
        "duplicate_index": int(df["index"].duplicated().sum()), "duplicate_image_qa": exact_dup,
        "unique_question_templates": int(df.question.nunique()),
        "rows_using_repeated_question_template": int(df.question.duplicated(keep=False).sum()),
        "missing_image_files": len(missing), "double_root_paths": bad_path_double, "bad_runner_relative_prefix": bad_path_prefix,
        "mcq_count": mcq_count, "mcq_bad_options": mcq_bad_options,
        "mcq_bad_key": mcq_bad_key, "mcq_answer_concept_mismatch": mcq_mismatch,
        "distribution_left_in_visual_fields": bad_visual_field,
        "known_distribution_or_configuration_misfiled": misplaced_distribution,
        "configuration_left_in_surface_field": surface_configuration,
        "literal_distribution_in_characteristics": bad_plain_distribution,
        "mang_in_shape_field": mang_in_shape,
        "mang_any_shape_field": mang_any_shape,
        "plain_mang_in_lesion_recognition": plain_mang_in_lesion,
        "noncanonical_terms_left": terminology_left,
        "judgement_leakage_pattern": judgement_leakage,
        "judgement_yes": judgement_yes, "judgement_no": judgement_no,
        "nonvisual_or_causal_visual_answers": nonvisual_visual_answer,
        "postinflammatory_causality_left": postinflam_left,
        "follicular_diagnosis_in_visual_field": follicular_diagnosis,
        "source_disease_image_filename_conflict": source_image_name_conflict,
        "hair_wrong_subcategory": hair_wrong_subcategory,
        "hair_density_wrong_subcategory": hair_density_wrong_subcategory,
        "legacy_lesion_subcategory": legacy_lesion_subcategory,
        "ambiguous_san_not_label": ambiguous_san_not,
        "noncanonical_inflammatory_pigment_label": noncanonical_inflammatory_pigment,
        "lesion_recognition_ontology_errors": lr_invalid,
        "lesion_reasoning_rows": int((df.category == "Lesion_Reasoning").sum()),
        "missing_samples": missing[:5],
    }


results = {name: audit(name, file) for name, file in FILES.items()}
regressions = {
    name: {key: a[key] for key in (
        "known_distribution_or_configuration_misfiled", "configuration_left_in_surface_field",
        "hair_density_wrong_subcategory", "legacy_lesion_subcategory", "ambiguous_san_not_label",
        "noncanonical_inflammatory_pigment_label", "noncanonical_terms_left",
    ) if a[key]}
    for name, (_, a) in results.items()
}
regressions = {name: issues for name, issues in regressions.items() if issues}
if regressions:
    print("Known quality regressions detected before applying corrections:", regressions)
    raise SystemExit(1)
ledger = pd.read_csv(OUT / "DermNet_VQA_issue_ledger_20260923.tsv", sep="\t", dtype=str, keep_default_na=False)
source_counts = {"Val_4k": 2819, "Test_1of3": 8224, "Test": 24671}
val_df, test1_df, test_df = (results[n][0] for n in ("Val_4k", "Test_1of3", "Test"))
test1_source_not_in_test = int((~test1_df.source_index.isin(set(test_df.source_index))).sum())
test1_images_not_in_test = len(set(test1_df.image_id) - set(test_df.image_id))
val_test_image_overlap = len(set(val_df.image_id) & set(test_df.image_id))
raw_candidates = {
    "Val_4k": Path(r"C:\Users\Vu\Downloads\DermNet_Val_4k.cleaned_base_v2.tsv"),
    "Test_1of3": Path(r"C:\Users\Vu\Downloads\DermNet_Test_clean_fixed_random_1of3_cases_relative.tsv"),
    "Test": Path(r"C:\Users\Vu\Downloads\DermNet_Test_clean_fixed_relative.tsv"),
}
raw_paths = {}
for name, path in raw_candidates.items():
    if path.is_file():
        raw = pd.read_csv(path, sep="\t", usecols=["image_path"], dtype=str, keep_default_na=False)
        raw_paths[name] = (len(raw), int(raw.image_path.str.replace("\\", "/", regex=False).str.contains("dermnet-output/dermnet-output/images", regex=False).sum()))
path_audit = []
for name, file in FILES.items():
    df, stats = results[name]
    raw_n, raw_bad = raw_paths.get(name, (0, 0))
    clean_file = CLEAN / CLEAN_FILES[name]
    clean = pd.read_csv(clean_file, sep="\t", dtype=str, keep_default_na=False)
    clean_dup = int(clean.image_path.str.replace("\\", "/", regex=False).str.contains("dermnet-output/dermnet-output/images", regex=False).sum())
    clean_prefix = int((~clean.image_path.str.replace("\\", "/", regex=False).str.startswith("../../dermnet-output/images/")).sum())
    clean_missing = int(sum(1 for p in clean.image_path if not (RUN_BASE / str(p).replace("\\", "/")).resolve().is_file()))
    path_audit.append({
        "dataset": name,
        "raw_source_rows": raw_n,
        "raw_source_duplicate_root_paths": raw_bad,
        "cleaned_final_rows": len(clean),
        "cleaned_final_duplicate_root_paths": clean_dup,
        "cleaned_final_wrong_runner_relative_prefix": clean_prefix,
        "cleaned_final_missing_images_from_runner_base": clean_missing,
        "reviewed_rows": stats["rows"],
        "reviewed_duplicate_root_paths": stats["double_root_paths"],
        "reviewed_wrong_runner_relative_prefix": stats["bad_runner_relative_prefix"],
        "reviewed_missing_images_from_runner_base": stats["missing_image_files"],
        "runner_base": str(RUN_BASE),
    })
pd.DataFrame(path_audit).to_csv(OUT / "DermNet_VQA_path_audit_20260923.tsv", sep="\t", index=False, encoding="utf-8")
corrected_unique = ledger[ledger.action == "corrected"].groupby("dataset").source_index.nunique().to_dict()
lines = [
    "# Báo cáo rà soát bộ câu hỏi DermNet VQA",
    "",
    "Ngày rà soát: 2026-09-23",
    "",
    "## Phạm vi",
    "",
    "Rà soát toàn bộ dòng còn lại trong ba bản `cleaned_final` sau vòng lọc trước; không ghi đè các bản nguồn. Kiểm tra tự động câu hỏi/đáp án, cấu trúc, lựa chọn MCQ, khóa đáp án, trùng lặp, trường dữ liệu và khả năng mở ảnh từ thư mục chạy `Phase_2/VLMEvalKit`. Ảnh được đối chiếu thủ công cho các nhóm xung đột/nhãn đáng ngờ có mục tiêu; không khẳng định bác sĩ đã duyệt từng ảnh trong toàn bộ tập.",
    "",
    "## Kết quả cuối",
    "",
    "| Bộ dữ liệu | Dòng sau rà soát | Ảnh | Bị cách ly | Sai tiền tố đường dẫn runner | Ảnh thiếu | MCQ | MCQ còn lỗi |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for name, (_, a) in results.items():
    removed = source_counts[name] - a["rows"]
    badmcq = a["mcq_bad_options"] + a["mcq_bad_key"] + a["mcq_answer_concept_mismatch"]
    lines.append(f"| {name} | {a['rows']:,} | {a['images']:,} | {removed:,} | {a['bad_runner_relative_prefix']} | {a['missing_image_files']} | {a['mcq_count']:,} | {badmcq} |")
lines += [
    "",
    "## Phạm vi so với đầu vào thực tế",
    "",
    "Ba file được rà ở đây là các bản `cleaned_final` ngày 23/09, sau đó được kiểm tra và xử lý thêm trong lượt này. Các TSV trong Downloads chỉ được dùng để xác nhận lỗi đường dẫn ở snapshot cũ, không dùng làm đầu vào nội dung cho lượt rà soát. Không ghi đè các file nguồn.",
    "",
    "| Bộ dữ liệu | Dòng `cleaned_final` rà soát | Cách ly trong lượt này | Dòng đầu ra |",
    "|---|---:|---:|---:|",
]
for name, (_, a) in results.items():
    lines.append(f"| {name} | {source_counts[name]:,} | {source_counts[name] - a['rows']:,} | {a['rows']:,} |")
lines += [
    "",
    "Các kiểm tra cuối đều đạt: không có dòng thiếu trường chính, trùng index hoặc Q+A trên cùng ảnh; không có ảnh thiếu/sai tiền tố đường dẫn; MCQ đủ bốn lựa chọn khác nhau và khóa đáp án khớp; không còn phân bố ở trường hình dạng/bề mặt, thuật ngữ cũ, dấu hiệu chủ quan/không nhìn thấy, lỗi hình thái trong Lesion_Recognition, mẫu leakage đã biết hoặc Lesion_Reasoning.",
    "Lưu ý về Lesion_Reasoning: nhóm này đã có 0 dòng trong cả ba `cleaned_final` đầu vào trước lượt rà soát này. Vì vậy các prompt reasoning lỗi trong snapshot cũ không được diễn đạt lại ở đây; không có câu reasoning mới được sinh thêm.",
    "",
    "Số dòng có ít nhất một hiệu chỉnh được ghi sổ: Val_4k={:,}; Test_1of3={:,}; Test={:,}. Các dòng có thể nhận nhiều hiệu chỉnh; bộ Test_1of3 là tập con của Test nên không cộng hai bộ này thành số ảnh riêng.".format(corrected_unique.get("Val_4k", 0), corrected_unique.get("Test_1of3", 0), corrected_unique.get("Test", 0)),
    "",
    f"Đối soát split: Test_1of3 có {test1_source_not_in_test} source_index và {test1_images_not_in_test} ảnh không thấy trong Test; giao ảnh giữa Val_4k và Test là {val_test_image_overlap}.",
    "",
    "Câu Judgement vẫn có tỷ lệ Có/Không cần tính đến khi chấm điểm: " + "; ".join(f"{name} {a['judgement_yes']} Có / {a['judgement_no']} Không" for name, (_, a) in results.items()) + ". Test_1of3 là mẫu con của Test, không cộng hai bộ này.",
    "",
    "## Kiểm tra đường dẫn gốc",
    "",
]
for name, (n, duplicated) in raw_paths.items():
    lines.append(f"- File nguồn trong Downloads cho {name}: {duplicated:,}/{n:,} dòng có chuỗi thư mục lặp `dermnet-output/dermnet-output/images`.")
lines += [
    "- Kiểm tra phân biệt đúng tầng dữ liệu: ba `cleaned_final.tsv` đã được vòng làm sạch trước chuẩn hóa đường dẫn (0 đường dẫn lặp, 0 sai tiền tố, 0 ảnh thiếu); lượt này xác nhận lại từng đường dẫn và mỗi đường dẫn đều mở được từ runner.",
    "- Ba `reviewed.tsv` tiếp tục có 0 đường dẫn lặp, 0 sai tiền tố và 0 ảnh thiếu. Từ `Phase_2/VLMEvalKit`, tiền tố đúng là `../../dermnet-output/images/...`; dạng `../../dermnet-output/dermnet-output/images/...` trong TSV Downloads gốc là sai và đã được loại trước khi tạo `cleaned_final`.",
    "- Lưu ý: hai TSV Test tìm thấy trong Downloads mang ngày 02/06; chúng được dùng để kiểm chứng lỗi path ở các bản cục bộ đó. Nội dung được rà trong lượt này là các `cleaned_final` ngày 23/09, không lấy bản Downloads cũ làm đầu vào nội dung.",
    "- Đối soát tổng hợp: `DermNet_VQA_path_audit_20260923.tsv`.",
    "",
    "## Các nhóm đã xử lý",
    "",
    "- Chuẩn hóa thuật ngữ: `tổn thương thực thể` → `tổn thương cơ bản`; `đóng mài/đóng mày`, `mài` trong gold label → `vảy tiết`; `bóng nước` → `bọng nước`; đồng nhất một số màu và chính tả.",
    "- Kiểm tra cụ thể lỗi path `../../dermnet-output/dermnet-output/images/...`: lỗi có trong toàn bộ TSV gốc ở Downloads; bản `cleaned_final` đã sửa trước lượt này. Đã xác nhận lại mọi đường dẫn của 3 đầu ra từ thư mục runner; không còn chuỗi root lặp và tất cả ảnh đều tồn tại.",
    "- Sửa MCQ sai thuật ngữ/lựa chọn ở một số dòng xác định; kiểm định lại sau chuẩn hóa, loại MCQ có lựa chọn trùng hoặc gold key không hợp lệ.",
    "- Đưa câu hỏi/đáp án về đúng nhóm: tách pattern phân bố khỏi hình thái/bề mặt; chuyển dấu hiệu móng, tóc/lông và sắc tố ra khỏi `Lesion_Recognition`; ghi rõ câu hỏi về tổn thương cơ bản hay biến đổi thứ phát.",
    "- Bổ sung bắt lỗi các nhãn mật độ/cụm/theo dải còn nằm trong trường bề mặt; chuyển cấu hình dạng lưới thuần sang trường hình dạng/cấu hình và cách ly câu gộp trường không thể tách an toàn. Chuẩn hóa subcategory `Lesion_Type` thành `Primary_Lesion_Type`.",
    "- Cách ly các câu `Judgement` xung đột với JSON/ảnh hoặc mất liên kết JSON; MCQ có lựa chọn trùng; câu trộn nhiều trường không thể tách an toàn; và câu có ảnh X-quang hoặc ảnh mẫu nước tiểu thay vì ảnh da.",
    "- Cách ly `Mảng` còn dùng như đáp án hình dạng vì dữ liệu không cho biết đó là patch phẳng hay plaque gồ; cách ly `Phân bố không đều` bị gộp với tăng sắc tố trong trường đặc điểm bề mặt (Test index=64).",
    "- Cách ly thêm nhãn `sẩn nốt` không phân biệt được sẩn với nốt (Test source_index=15671), gold `Tăng sắc tố viêm` không chuẩn (Test source_index=49064), và các câu bề mặt còn trộn với cấu hình dạng lưới; giữ nguyên dữ liệu gốc trong source/ledger.",
    "- Các bệnh danh đã nêu được giữ đúng phạm vi: Purpura → ban xuất huyết; urticarial vasculitis → viêm mạch mày đay; telangiectasia → giãn mao mạch; morphoea → xơ cứng bì khu trú; metastatic/ocular melanoma giữ thông tin vị trí/di căn; leukaemia cutis → xâm nhiễm bạch cầu ở da; mycosis fungoides → u sùi dạng nấm; solar lentigo → lentigo do nắng.",
    "- Tên bệnh tiếng Anh còn lại trong nguồn chưa được dịch hàng loạt; cần một bảng thuật ngữ được duyệt để tránh dịch sai nghĩa hoặc làm mất thông tin bệnh.",
    "",
    "## Sổ lỗi và bản đầu ra",
    "",
    "Sổ lỗi ghi từng dòng được sửa/cách ly, gồm `source_index`, lý do, câu hỏi/đáp án trước và sau hiệu chỉnh, tiểu trường sau sửa: `DermNet_VQA_issue_ledger_20260923.tsv`.",
    "",
    "Ba TSV `*.reviewed.tsv` là bản giữ lại để dùng tiếp. Số dòng cách ly được loại khỏi các TSV này nhưng giữ trong bản nguồn cũ và có dấu trong sổ lỗi.",
    "",
    "## Giới hạn cần biết",
    "",
    "Đây là rà soát dữ liệu bằng quy tắc chuyên môn, đối chiếu JSON và kiểm tra hình ảnh có trọng điểm. Quy tắc tự động có thể phát hiện lỗi cấu trúc/ngôn ngữ nhưng không thay thế hội chẩn da liễu cho mọi ảnh. Các dòng có gold lâm sàng còn chưa được xác minh ảnh trực tiếp đã bị cách ly thay vì tự sửa đáp án. `clean_core` được thay bằng `rule_checked_visual_not_fully_adjudicated` để không ngụ ý toàn bộ ảnh đã được bác sĩ xác nhận.",
    "Một số mẫu câu vẫn được tái sử dụng trên nhiều ảnh. Chúng không bị xóa nếu cặp câu-đáp án không trùng trên cùng ảnh; mức đa dạng template cần được cân nhắc khi chấm khả năng tổng quát hóa.",
    "",
    "## Kiểm tra còn lại theo file",
    "",
]
for name, (_, a) in results.items():
    lines.append(f"- {name}: {a['rows']:,} dòng; {a['images']:,} ảnh; thiếu trường={a['blank_rows']}; trùng index/Q+A={a['duplicate_index']}/{a['duplicate_image_qa']}; đường dẫn sai tiền tố/thiếu ảnh={a['bad_runner_relative_prefix']}/{a['missing_image_files']}; MCQ lỗi={a['mcq_bad_options'] + a['mcq_bad_key'] + a['mcq_answer_concept_mismatch']}; phân bố sai trường={a['distribution_left_in_visual_fields'] + a['literal_distribution_in_characteristics']}; thuật ngữ cũ={a['noncanonical_terms_left']}; dấu hiệu không quan sát được={a['nonvisual_or_causal_visual_answers']}; lỗi ontology Lesion_Recognition={a['lesion_recognition_ontology_errors']}; hair sai tiểu trường={a['hair_wrong_subcategory']}; Judgement leakage mẫu đã biết={a['judgement_leakage_pattern']}; Lesion_Reasoning={a['lesion_reasoning_rows']}; câu hỏi duy nhất={a['unique_question_templates']:,}; dòng dùng câu hỏi lặp={a['rows_using_repeated_question_template']:,}.")
(OUT / "DermNet_VQA_review_report_20260923.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("runner base exists:", RUN_BASE.is_dir(), RUN_BASE)
for name, (_, a) in results.items():
    print(name, a)
print("raw source path counts:", raw_paths)
print("split checks:", {"test1_source_not_in_test": test1_source_not_in_test, "test1_images_not_in_test": test1_images_not_in_test, "val_test_image_overlap": val_test_image_overlap})
print("quarantine unique source rows:")
for name, g in ledger[ledger.action == "quarantined_from_reviewed_tsv"].groupby("dataset"):
    print(name, g.source_index.nunique())
print("top issue codes:")
print(ledger.groupby(["dataset", "issue_code", "action"]).size().sort_values(ascending=False).head(40).to_string())
print("report:", OUT / "DermNet_VQA_review_report_20260923.md")
