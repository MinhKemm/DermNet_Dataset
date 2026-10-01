from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\VuLapTrinh2\DermNet_Dataset")
SRC = ROOT / "outputs" / "dermnet-vqa-cleaned-20260923"
OUT = ROOT / "outputs" / "dermnet-vqa-reviewed-20260923"
OUT.mkdir(parents=True, exist_ok=True)
FILES = {
    "Val_4k": ("DermNet_Val_4k.cleaned_final.tsv", "DermNet_Val_4k.reviewed.tsv"),
    "Test_1of3": ("DermNet_Test_1of3.cleaned_final.tsv", "DermNet_Test_1of3.reviewed.tsv"),
    "Test": ("DermNet_Test.cleaned_final.tsv", "DermNet_Test.reviewed.tsv"),
}
frames = {k: pd.read_csv(SRC / a, sep="\t", dtype=str, keep_default_na=False) for k, (a, _) in FILES.items()}
quarantine: dict[str, set[str]] = defaultdict(set)
issues: list[dict[str, str]] = []


def jl(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def concepts(value):
    try:
        x = json.loads(value)
        return x if isinstance(x, list) else []
    except Exception:
        return []


def options(question):
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"(?m)^\s*([A-D])\.\s*(.*?)\s*$", question)}


def letters(answer):
    return "".join(c for c in str(answer).upper() if c in "ABCD")


def format_vi(xs):
    xs = [str(x).strip() for x in xs if str(x).strip()]
    if len(xs) < 2:
        return xs[0] if xs else ""
    if len(xs) == 2:
        return f"{xs[0]} và {xs[1]}"
    return ", ".join(xs[:-1]) + f" và {xs[-1]}"


def add_issue(dataset, row, code, action, detail, q_before=None, a_before=None):
    issues.append({
        "dataset": dataset, "index": row.get("index", ""), "source_index": row.get("source_index", ""),
        "image_id": row.get("image_id", ""), "source_disease": row.get("source_disease", ""),
        "category_after": row.get("category", ""), "sub_category_after": row.get("sub_category", ""),
        "type": row.get("type", ""),
        "issue_code": code, "action": action, "detail": detail, "image_path": row.get("image_path", ""),
        "question_before": row.get("question", "") if q_before is None else q_before,
        "answer_before": row.get("answer", "") if a_before is None else a_before,
        "question_after": row.get("question", ""), "answer_after": row.get("answer", ""),
        "answer_concepts_after": row.get("answer_concepts", ""),
    })


def add_transform(df, i, tag):
    x = concepts(df.at[i, "transformations"])
    if tag not in x:
        x.append(tag)
    df.at[i, "transformations"] = jl(x)


def quarantine_row(name, i, code, detail):
    row = frames[name].loc[i].to_dict()
    quarantine[name].add(str(row.get("source_index", "")))
    add_issue(name, row, code, "quarantined_from_reviewed_tsv", detail)


def replace_ci(text, old, new):
    return re.sub(re.escape(old), lambda _: new, str(text), flags=re.I)


def normalize_text(s):
    for old, new in [
        ("tổn thương thực thể", "tổn thương cơ bản"),
        ("viềng hồng ban xung quanh", "viền hồng ban xung quanh"),
        ("vảy mài", "vảy tiết"),
        ("đóng mài", "vảy tiết"), ("đóng mày", "vảy tiết"),
        ("bóng nước", "bọng nước"), ("oval", "bầu dục"),
        ("tăng sắc tố sau viêm", "tăng sắc tố"),
        ("thâm sau viêm", "tăng sắc tố"),
        ("tăng sắc tố hậu viêm", "tăng sắc tố"),
        ("xâm nhiễm tế bào bạch cầu ở da", "xâm nhiễm bạch cầu ở da"),
    ]:
        s = replace_ci(s, old, new)
    s = re.sub(r"\bmài\b", "vảy tiết", s, flags=re.I)
    return s


COLOR_MAP = {
    "hồng đỏ": "Đỏ hồng", "đỏ hồng": "Đỏ hồng", "nâu đỏ": "Đỏ nâu", "đỏ nâu": "Đỏ nâu",
    "nâu vàng": "Vàng nâu", "vàng nâu": "Vàng nâu", "vàng trắng": "Trắng vàng", "trắng vàng": "Trắng vàng",
    "xám trắng": "Trắng xám", "trắng xám": "Trắng xám",
}


def canon(x):
    x = normalize_text(str(x)).strip()
    if x.casefold() == "mài":
        x = "Vảy tiết"
    if x.casefold() in COLOR_MAP:
        return COLOR_MAP[x.casefold()]
    if x and x.casefold() not in {"có", "không"}:
        x = x[0].upper() + x[1:]
    return x


def canon_label(x):
    """Canonicalize short gold/option labels, including compound color labels."""
    x = normalize_text(str(x)).strip()
    if x.casefold() == "mài":
        x = "Vảy tiết"
    for old, new in COLOR_MAP.items():
        x = replace_ci(x, old, new)
    if x and x.casefold() not in {"có", "không"}:
        x = x[0].upper() + x[1:]
    return x


def set_mcq(df, i, opts, answer=None):
    if answer is not None:
        df.at[i, "answer"] = answer
    lines = df.at[i, "question"].splitlines()
    header = []
    for line in lines:
        if re.match(r"^\s*[A-D]\.\s*", line):
            break
        header.append(line)
    df.at[i, "question"] = "\n".join(header + [f"{k}. {opts[k]}" for k in "ABCD" if k in opts])
    df.at[i, "answer_concepts"] = jl([opts[c] for c in letters(df.at[i, "answer"]) if c in opts])


def mark_fix(name, df, i, code, detail, oldq, olda, tag):
    add_transform(df, i, tag)
    row = df.loc[i].to_dict()
    add_issue(name, row, code, "corrected", detail, oldq, olda)


# Normalize relative paths and text, then align answer_concepts with the corrected MCQ options.
for name, df in frames.items():
    for i in df.index:
        oldq, olda = df.at[i, "question"], df.at[i, "answer"]
        df.at[i, "image_path"] = str(df.at[i, "image_path"]).replace("\\", "/")
        df.at[i, "question"] = normalize_text(oldq)
        df.at[i, "answer"] = normalize_text(olda)
        opts = options(df.at[i, "question"])
        if df.at[i, "type"] == "Multi_choice" and len(opts) == 4:
            opts = {k: canon_label(v) for k, v in opts.items()}
            set_mcq(df, i, opts)
        else:
            vals = list(dict.fromkeys(canon(x) for x in concepts(df.at[i, "answer_concepts"])))
            df.at[i, "answer_concepts"] = jl(vals)
            if vals and df.at[i, "type"] not in ("Judgement", "Multi_choice"):
                df.at[i, "answer"] = format_vi(vals)
        p = df.at[i, "image_path"]
        fixed = p.replace("../../dermnet-output/dermnet-output/images/", "../../dermnet-output/images/")
        if fixed != p:
            df.at[i, "image_path"] = fixed
            add_transform(df, i, "path_duplicate_root_removed")
        if oldq != df.at[i, "question"] or olda != df.at[i, "answer"]:
            mark_fix(name, df, i, "terminology_canonicalized", "Chuẩn hóa chính tả/thuật ngữ; không đổi nhãn bệnh nguồn.", oldq, olda, "canonicalized_dermatology_terms")

# Targeted clinical corrections retained from source/image review and authoritative terminology.
for name, df in frames.items():
    for i in df.index:
        ix, disease = str(df.at[i, "index"]), df.at[i, "source_disease"]
        if ix == "0" and disease == "Acanthoma fissuratum" and df.at[i, "type"] == "Multi_choice":
            oldq, olda = df.at[i, "question"], df.at[i, "answer"]
            df.at[i, "question"] = "Trong ảnh được cung cấp, loại tổn thương cơ bản nào xuất hiện?\nChọn một đáp án đúng.\nA. Nốt hồng ban\nB. Sẩn phù\nC. Mụn nước\nD. Mụn mủ"
            df.at[i, "answer"] = "A"
            df.at[i, "answer_concepts"] = jl(["Nốt hồng ban"])
            df.at[i, "scoring_method"] = "exact_option"
            mark_fix(name, df, i, "nonstandard_U_mun_gold", "Bỏ gold ‘U mụn’; giữ nhãn nốt được nguồn/ảnh hỗ trợ và thay lựa chọn bằng các loại tổn thương cùng cấp.", oldq, olda, "U_mun_removed_and_parallel_morphology_mcq")
        if ix == "3" and disease == "Acanthoma fissuratum" and df.at[i, "type"] == "Multi_choice":
            oldq, olda = df.at[i, "question"], df.at[i, "answer"]
            op = options(df.at[i, "question"])
            if len(op) == 4:
                op["B"], op["C"] = "Rãnh trung tâm", "Viền hồng ban xung quanh"
                set_mcq(df, i, op, "BCD")
                mark_fix(name, df, i, "central_furrow_and_spelling", "Đổi ‘Mụn nhỏ trung tâm’ thành rãnh trung tâm theo mô tả/ảnh; sửa ‘Viềng’ thành ‘Viền’.", oldq, olda, "acanthoma_fissuratum_feature_fix")
        if ix == "17" and disease == "Acanthosis palmaris" and df.at[i, "category"] == "Attribute_Characteristics":
            oldq, olda = df.at[i, "question"], df.at[i, "answer"]
            vals = ["Da dày", "Vân da nổi rõ"]
            df.at[i, "answer_concepts"], df.at[i, "answer"] = jl(vals), format_vi(vals)
            mark_fix(name, df, i, "tripe_palms_translation", "Sửa thành ‘Da dày, Vân da nổi rõ’; bỏ cụm dịch sai về mạch máu và chi tiết cắt móng không phải dấu hiệu tổn thương.", oldq, olda, "tripe_palms_features_corrected")
        if ix == "44" and disease == "Acne vulgaris" and df.at[i, "type"] == "Multi_choice":
            oldq, olda = df.at[i, "question"], df.at[i, "answer"]
            op = {"A": "Bàn chân; phân bố khu trú", "B": "Lưng trên và vai; phân bố hai bên, rải rác", "C": "Bàn chân; phân bố một bên", "D": "Bàn chân; phân bố đối xứng"}
            df.at[i, "question"] = "Dựa vào hình ảnh trên, phương án nào mô tả đúng vị trí giải phẫu và kiểu phân bố của tổn thương?\n" + "\n".join(f"{k}. {v}" for k, v in op.items())
            df.at[i, "answer"], df.at[i, "answer_concepts"] = "B", jl([op["B"]])
            mark_fix(name, df, i, "anatomy_distribution_answer", "Tách ‘Vai’ thành vị trí giải phẫu; phân bố ghi riêng là hai bên và rải rác.", oldq, olda, "anatomy_distribution_terms_split")
        if ix == "5212" and disease == "Anagen effluvium" and df.at[i, "category"] == "Attribute_Color":
            oldq, olda = df.at[i, "question"], df.at[i, "answer"]
            df.at[i, "category"], df.at[i, "sub_category"] = "Anatomical_Distribution", "Distribution"
            df.at[i, "question"], df.at[i, "answer"], df.at[i, "answer_concepts"] = "Trong hình ảnh này, vị trí giải phẫu của tổn thương là gì?", "Da đầu", jl(["Da đầu"])
            mark_fix(name, df, i, "anatomy_in_color_field", "‘Da đầu’ là vị trí giải phẫu, không phải màu sắc.", oldq, olda, "anatomical_location_reclassified")
        if ix == "46916" and disease == "Radiation dermatitis":
            oldq, olda = df.at[i, "question"], df.at[i, "answer"]
            df.at[i, "question"] = "Trong ảnh, có xác định được vị trí giải phẫu không và tổn thương phân bố như thế nào?"
            df.at[i, "answer"] = "Không xác định được vị trí giải phẫu; tổn thương khu trú."
            df.at[i, "answer_concepts"] = jl(["Không xác định được vị trí giải phẫu", "Khu trú"])
            mark_fix(name, df, i, "unknown_anatomy_clarified", "Giữ thông tin thiếu mốc giải phẫu và pattern khu trú; bỏ ‘Vùng da’ quá chung.", oldq, olda, "anatomy_unknown_and_distribution_localized_separated")
        if disease == "Epidermolysis bullosa acquisita" and str(df.at[i, "source_index"]) == "15671":
            quarantine_row(name, i, "ambiguous_papulonodular_label", "Cụm ‘sẩn nốt’ không phân biệt rõ sẩn với nốt hoặc hình thái kết hợp; ảnh không đủ để tự chuẩn hóa nhãn gold.")
        if disease == "Scurvy" and str(df.at[i, "source_index"]) == "49064":
            quarantine_row(name, i, "noncanonical_inflammatory_pigment_label", "‘Tăng sắc tố viêm’ không phải mô tả thị giác chuẩn, còn hàm ý nguyên nhân; cách ly cả QA cho tới khi nhãn màu được bác sĩ xác nhận.")
        if disease == "Trichotillomania" and str(df.at[i, "source_index"]) == "54339" and df.at[i, "category"] == "Attribute_Characteristics":
            oldq, olda = df.at[i, "question"], df.at[i, "answer"]
            df.at[i, "sub_category"] = "Hair_and_Surface_Characteristics"
            opts = options(oldq)
            if len(opts) == 4:
                df.at[i, "question"] = "Trong ảnh, dấu hiệu tóc/lông hoặc bề mặt da nào được quan sát?\nChọn tất cả đáp án đúng.\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())
            mark_fix(name, df, i, "hair_and_surface_mcq_scope_corrected", "Gold chọn dấu hiệu tóc/lông nhưng các lựa chọn nhiễu thuộc bề mặt da; làm rõ câu hỏi hỗn hợp và gán đúng tiểu trường.", oldq, olda, "hair_and_surface_scope_corrected")
        if df.at[i, "category"] == "Lesion_Recognition" and df.at[i, "sub_category"] == "Lesion_Type":
            oldq, olda = df.at[i, "question"], df.at[i, "answer"]
            df.at[i, "sub_category"] = "Primary_Lesion_Type"
            mark_fix(name, df, i, "legacy_lesion_subcategory", "Đổi tên tiểu trường cũ Lesion_Type sang Primary_Lesion_Type để thống nhất ontology.", oldq, olda, "primary_lesion_subcategory_canonicalized")

# Image/JSON judgement audit: indices below are row positions in the old cleaned TSV.
val_conflicts = [47, 144, 451, 1352, 1871, 2455]
val_unmatched = [1640, 2244, 2330, 2391, 2418]
test_conflicts = [203,2022,11404,12471,12623,12838,13559,21812,1452,1481,3488,5343,18884,19459,19968,1464,1514,4511,5972,9782,10937,11020,11089,11236,11383,11402,12294,12424,12445,12683,13318,13328,13342,13795,13956,14002,14011,14662,14804,14870,14937,15269,15464,15506,16045,16074,16282,16342,16477,16830,17066,17344,17624,18244,18333,18461,18549,18836,18875,18956,19259,19483,19508,19692,19724,19941,20291,20335,20380,20530,21142,21224,21557,22174,22340,22411,22492,23782,24120,24272,24343,24639,24648,3564,4287,7300]
test_unmatched = [14375,14376,14585,15438,15439,15821,16402,16405,16913,17153,18457,20185,22308,22803,24478]
for pos in val_conflicts:
    quarantine_row("Val_4k", pos, "judgement_conflicts_with_json_or_image", "Đáp án Có/Không mâu thuẫn với JSON, ảnh hoặc chưa xác minh đủ; cần adjudication trước khi dùng làm gold.")
for pos in val_unmatched:
    quarantine_row("Val_4k", pos, "judgement_source_json_unmatched", "Không tìm được JSON nguồn tương ứng; cách ly cho đến khi khôi phục liên kết nguồn.")
for pos in test_conflicts:
    quarantine_row("Test", pos, "judgement_conflicts_with_json_or_image", "Đáp án Có/Không mâu thuẫn với JSON, ảnh hoặc chưa xác minh đủ; cần adjudication trước khi dùng làm gold.")
for pos in test_unmatched:
    quarantine_row("Test", pos, "judgement_source_json_unmatched", "Không tìm được JSON nguồn tương ứng; cách ly cho đến khi khôi phục liên kết nguồn.")
test_reasons = {}
for pos in test_conflicts + test_unmatched:
    row = frames["Test"].iloc[pos]
    test_reasons[str(row.source_index)] = "judgement_conflicts_with_json_or_image" if pos in test_conflicts else "judgement_source_json_unmatched"
for i, row in frames["Test_1of3"].iterrows():
    code = test_reasons.get(str(row.source_index))
    if code:
        quarantine_row("Test_1of3", i, code, "Dòng thuộc tập Test 1/3 và cùng nguồn với một dòng Test đang chờ adjudication.")

# Exclude the two Porphyria cutanea tarda sample images: urine/cups, not skin lesions.
for name, df in frames.items():
    if name not in ("Test", "Test_1of3"):
        continue
    for i, r in df.iterrows():
        if r.source_disease == "Porphyria cutanea tarda" and str(r["index"]) in {"42395","42396","42397","42398","42407","42408","42409","42410"}:
            quarantine_row(name, i, "image_is_not_a_skin_lesion", "Ảnh là mẫu nước tiểu/ống nghiệm, không phải tổn thương da; loại toàn bộ QA gắn với ảnh.")
        if r.category == "Lesion_Recognition" and str(r["index"]) == "34" and r.source_disease == "Acne vulgaris":
            quarantine_row(name, i, "ambiguous_fibrous_papule_gold", "‘Sẩn sợi’ không được hỗ trợ bởi ảnh Acne vulgaris 142; không tự đoán thuật ngữ thay thế.")
        if str(r["index"]) == "51672" and r.source_disease == "Systemic sclerosis":
            quarantine_row(name, i, "radiology_image_not_dermatology_photo", "Ảnh là X-quang bàn tay; câu có thuật ngữ ‘cản quang’ không mô tả đặc điểm da. Loại khỏi VQA ảnh da.")

# Quarantine rows whose lesion-type gold is visibly mixed with non-morphology,
# whose tactile evidence cannot be judged from a photograph, or whose source
# disease and image filename conflict. Keep all other QA for a mismatched image
# out of the reviewed benchmark as well.
known_unverified_primary = {
    "Val_4k": {"979"},
    "Test_1of3": {"18573"},
    "Test": {"14641", "18541", "18566", "18573", "44915", "50304"},
}
for name, df in frames.items():
    for i, r in df.iterrows():
        if str(r.source_index) in known_unverified_primary.get(name, set()):
            quarantine_row(name, i, "non_primary_or_unverified_lesion_recognition_gold", "Câu hỏi yêu cầu loại tổn thương cơ bản nhưng gold trộn sưng nề/dày mô mềm/ban đỏ hoặc dấu quanh móng; ảnh chưa xác nhận được một hình thái cơ bản đủ chắc chắn.")
        if r.source_disease == "Digital myxoid pseudocyst" and str(r.source_index) == "13007":
            quarantine_row(name, i, "tactile_or_unverified_visual_evidence", "MCQ có ‘mô mềm’ là dấu hiệu sờ khám; ‘ướt/rớm máu’ cũng chưa được xác minh trên ảnh. Không dùng làm gold ảnh.")
        if r.source_disease == "Adult-onset Still disease" and "cutaneous-mastocytosis-" in str(r.image_path).casefold():
            same_image = df.index[df.image_id.astype(str).eq(str(r.image_id))]
            for j in same_image:
                quarantine_row(name, j, "source_disease_image_filename_conflict", "Tên bệnh nguồn Adult-onset Still disease không khớp tên ảnh cutaneous-mastocytosis; cách ly toàn bộ QA của ảnh đến khi khôi phục mapping đúng.")
        visual_fields = r.category != "Diagnosis"
        text_blob = " ".join([str(r.question), str(r.answer), str(r.answer_concepts)])
        if visual_fields and re.search(r"(?i)\b(?:hậu viêm|sau viêm|post[- ]?inflammatory)\b", text_blob):
            quarantine_row(name, i, "temporal_or_causal_inference_in_visual_answer", "Cụm ‘sau viêm/hậu viêm’ suy diễn nguyên nhân hoặc diễn tiến; chưa được chuẩn hóa thành dấu hiệu nhìn thấy rõ.")
        if visual_fields and re.search(r"(?i)\b(?:ngứa|đau|đau rát|rát|tenderness|itch(?:ing)?|pruritus|pain|palpable|sờ thấy|dễ chảy máu|bleeds? easily)\b", text_blob):
            quarantine_row(name, i, "nonvisual_symptom_or_palpatation_in_image_question", "Câu hỏi/đáp án dựa vào triệu chứng chủ quan, khả năng sờ thấy hoặc khuynh hướng chảy máu; ảnh tĩnh không xác nhận được.")
        if r.category == "Attribute_Characteristics" and re.search(r"(?i)\b(?:viêm nang lông|tổn thương nang lông)\b", text_blob):
            quarantine_row(name, i, "follicular_diagnosis_or_underspecified_label", "‘Viêm nang lông/tổn thương nang lông’ không mô tả một dấu hiệu hình ảnh đủ chuẩn hóa để dùng làm gold trong trường bề mặt.")

# Distribution terms in Shape/Characteristics are moved out of those fields or quarantined if a compound MCQ/Judgement cannot be split.
dist_terms = ["mật độ dày đặc","theo dải","tụ đám","hợp đám","kết đám","tụ cụm","dày đặc","khu trú","lan tỏa","lan rộng","rải rác","thưa thớt","thưa","đối xứng","hai bên","một bên","theo dermatome","vùng duỗi","vùng gấp","vùng kẽ","vùng tiếp xúc","vùng phơi nắng","vùng tiết bã","đa ổ","nhiều ổ","một ổ","đơn độc","đơn lẻ","thành cụm","tụ thành cụm","thành đám","tụ thành đám","cụm","hợp lưu","vệ tinh","tổn thương vệ tinh","tập trung","quanh nang lông","nhiều tổn thương"]
dist_re = re.compile(r"(?i)(?<!không )(?<!bất )\b(?:" + "|".join(re.escape(x) for x in dist_terms) + r")\b")

def is_hair_feature(s):
    x = re.sub(r"(?i)\b(?:quanh|xung quanh)\s+nang\s+lông\b", " ", str(s))
    x = re.sub(r"(?i)\bnang\s+lông\b", " ", x)
    return bool(re.search(r"(?i)\b(?:tóc|lông|chấy|trứng chấy|rụng tóc|alopecia)\b", x))

def distribution_search_text(s):
    # "Thưa tóc/lông" describes appendage density, not lesion distribution.
    return re.sub(r"(?i)\b(?:(?:thưa|thưa thớt)\s+(?:tóc|lông)|(?:tóc|lông)\s+(?:thưa|thưa thớt))\b", " ", str(s))

def is_pure_dist(s):
    # Asymmetry is a shape descriptor, not a distribution label.
    if is_hair_feature(s):
        return False
    if re.search(r"(?i)\b(?:không|bất)\s+đối xứng\b", str(s)):
        return False
    x = re.sub(r"(?i)\b(?:không|bất)\s+đối xứng\b", " ", str(s))
    x = re.sub(r"(?i)\bphân bố\b", " ", x)
    for term in sorted(dist_terms, key=len, reverse=True):
        x = re.sub(r"(?i)\b" + re.escape(term) + r"\b", " ", x)
    return not re.sub(r"[\s,;.+/()\-]+", "", x)

def strip_dist(xs):
    keep, removed = [], []
    for value in xs:
        parts = [x.strip() for x in re.split(r"(?i)\s*(?:,|;|\bvà\b)\s*", str(value)) if x.strip()]
        for part in parts:
            if is_hair_feature(part):
                keep.append(canon(part))
                continue
            if is_pure_dist(part):
                removed.append(part)
            else:
                cleaned = part
                for term in sorted(dist_terms, key=len, reverse=True):
                    cleaned = re.sub(r"(?i)(?<!không )(?<!bất )\b" + re.escape(term) + r"\b", " ", cleaned)
                cleaned = re.sub(r"(?i)\bphân bố\b", " ", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.-")
                if cleaned and cleaned.casefold() not in {"tổn thương", "đặc điểm", "phân bố", "kiểu"}:
                    keep.append(canon(cleaned))
                else:
                    removed.append(part)
    out=[]; seen=set()
    for x in keep:
        if x.casefold() not in seen:
            seen.add(x.casefold()); out.append(x)
    return out, removed

def claim(q):
    m = re.search(r'"([^"\n]+)"', str(q))
    if m: return m.group(1).strip()
    m = re.search(r"(?:tổn thương cơ bản chính có phải là|loại tổn thương cơ bản(?: chính)? là|tổn thương có đặc điểm lâm sàng)\s*(.*?)\s*(?:không\??|là gì\??|\?\s*$)", str(q), re.I | re.S)
    if m: return re.sub(r"\s+", " ", m.group(1)).strip(" .?\"'")
    m = re.search(r"(?:có đặc điểm(?: lâm sàng)?|đặc điểm lâm sàng)\s+(.*?)(?:\s+không\??|\?\s*$|\. Nhận định)", str(q), re.I | re.S)
    if m: return re.sub(r"\s+", " ", m.group(1)).strip(" .?\"'")
    m = re.search(r"\b(?:có|ghi nhận)\s+(.+?)\s+không\??\s*$", str(q), re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip(" .?\"'") if m else ""

for name, df in frames.items():
    for i, r in df.iterrows():
        if r.category not in ("Attribute_Shape", "Attribute_Characteristics"):
            continue
        opts = options(r.question) if r.type == "Multi_choice" else {}
        text = " ".join([r.question, r.answer, " ".join(map(str, concepts(r.answer_concepts))), " ".join(opts.values())])
        if not dist_re.search(distribution_search_text(text)):
            continue
        if r.type == "Multi_choice":
            quarantine_row(name, i, "distribution_mixed_into_shape_or_characteristics_mcq", "MCQ trộn đáp án hoặc lựa chọn về phân bố với hình thái/bề mặt; không thể giữ trường đơn nhất.")
            continue
        if r.type == "Judgement":
            target = claim(r.question)
            if target and is_pure_dist(target):
                oldq, olda = r.question, r.answer
                df.at[i,"category"], df.at[i,"sub_category"] = "Anatomical_Distribution", "Distribution"
                df.at[i,"question"] = f"Trong ảnh, tổn thương có kiểu phân bố {target} không?"
                add_transform(df,i,"distribution_judgement_reclassified")
                mark_fix(name,df,i,"distribution_judgement_wrong_category","Câu chỉ hỏi pattern phân bố; chuyển sang Anatomical_Distribution.",oldq,olda,"distribution_judgement_reclassified")
            else:
                quarantine_row(name,i,"compound_judgement_mixes_distribution_with_shape_or_surface","Câu Có/Không gộp pattern phân bố với hình thái/bề mặt nên không thể tách đáp án an toàn.")
            continue
        oldq, olda = r.question, r.answer
        vals, removed = strip_dist(concepts(r.answer_concepts))
        if not removed:
            continue
        if vals:
            df.at[i,"answer_concepts"] = jl(vals)
            df.at[i,"answer"] = format_vi(vals)
            add_transform(df,i,"distribution_terms_removed_from_visual_attribute_answer")
            mark_fix(name,df,i,"distribution_mixed_into_shape_or_characteristics_answer","Tách pattern phân bố khỏi đáp án hình thái/bề mặt.",oldq,olda,"distribution_terms_removed_from_visual_attribute_answer")
        else:
            has_dist_row = len(df[(df.image_id == r.image_id) & (df.category == "Anatomical_Distribution")]) > 0
            if has_dist_row:
                quarantine_row(name,i,"duplicate_distribution_question_in_wrong_category","Gold chỉ là pattern phân bố; cùng ảnh đã có câu Anatomical_Distribution.")
            else:
                df.at[i,"category"], df.at[i,"sub_category"] = "Anatomical_Distribution", "Distribution"
                df.at[i,"question"] = "Trong ảnh, kiểu phân bố của tổn thương là ____." if r.type == "Fill_in_blank" else "Trong ảnh, kiểu phân bố của tổn thương là gì?"
                add_transform(df,i,"distribution_question_reclassified")
                mark_fix(name,df,i,"distribution_answer_wrong_category","Gold chỉ mô tả phân bố; chuyển sang Anatomical_Distribution.",oldq,olda,"distribution_question_reclassified")

# Give hair/appendage and shape-configuration findings their own visual subfield.
shape_configuration = re.compile(r"(?i)\b(?:vân lưới|dạng lưới|vệt tuyến tính|tuyến tính|dạng vòng|hình vòng|dạng bia|hình bia|bia bắn|đồng tâm)\b")
for name, df in frames.items():
    for i, r in df.iterrows():
        if r.category == "Attribute_Characteristics":
            labels = [claim(r.question)] if r.type == "Judgement" else [str(x) for x in concepts(r.answer_concepts)]
            hair_labels = [x for x in labels if is_hair_feature(x)]
            if hair_labels:
                oldq, olda = r.question, r.answer
                other_labels = [x for x in labels if not is_hair_feature(x)]
                subcat = "Hair_and_Surface_Characteristics" if other_labels else "Hair_Morphology"
                df.at[i,"sub_category"] = subcat
                if r.type == "Judgement":
                    target = claim(oldq)
                    df.at[i,"question"] = f"Trong ảnh có dấu hiệu tóc/lông ‘{target}’ không?" if target else "Trong ảnh có dấu hiệu tóc/lông không?"
                elif r.type == "Multi_choice":
                    op = options(oldq)
                    header = "Đặc điểm tóc/lông và bề mặt da nào được ghi nhận?" if other_labels else "Dấu hiệu tóc/lông nào được quan sát trong ảnh?"
                    df.at[i,"question"] = header + "\nChọn tất cả đáp án đúng.\n" + "\n".join(f"{k}. {v}" for k, v in op.items())
                elif subcat == "Hair_Morphology":
                    df.at[i,"question"] = "Dấu hiệu tóc/lông được ghi nhận trong ảnh là ____ .".replace(" ____", "____") if r.type == "Fill_in_blank" else "Dấu hiệu tóc/lông nào được ghi nhận trong ảnh?"
                else:
                    df.at[i,"question"] = "Trong ảnh, những đặc điểm tóc/lông và bề mặt da nào được quan sát?" if r.type == "Short_answer" else "Trong ảnh, đặc điểm tóc/lông và bề mặt da là ____ .".replace(" ____", "____")
                mark_fix(name,df,i,"hair_feature_mislabeled_as_surface","Đưa dấu hiệu tóc/lông về tiểu trường tóc; giữ đặc điểm bề mặt đi kèm nếu có.",oldq,olda,"hair_feature_subcategory_corrected")
                continue
        if r.category == "Attribute_Characteristics":
            labels = concepts(r.answer_concepts)
            if labels and all(shape_configuration.search(str(x)) for x in labels):
                oldq, olda = r.question, r.answer
                df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Shape","Configuration"
                if r.type == "Judgement":
                    target=claim(oldq)
                    df.at[i,"question"] = f"Trong ảnh có cấu hình tổn thương dạng ‘{target}’ không?"
                elif r.type == "Multi_choice":
                    op=options(oldq)
                    df.at[i,"question"]="Cấu hình/hình dạng tổn thương nào được quan sát?\nChọn tất cả đáp án đúng.\n"+"\n".join(f"{k}. {v}" for k,v in op.items())
                elif r.type == "Fill_in_blank":
                    df.at[i,"question"]="Cấu hình tổn thương trong ảnh là ____ .".replace(" ____", "____")
                else:
                    df.at[i,"question"]="Cấu hình/hình dạng tổn thương trong ảnh là gì?"
                mark_fix(name,df,i,"configuration_mislabeled_as_surface","Đưa dạng lưới/tuyến tính/vòng/bia sang tiểu trường cấu hình hình thái.",oldq,olda,"shape_configuration_reclassified")

# A surface-field QA that still mixes configuration with texture, pigment, or
# secondary change cannot safely keep one field label. Quarantine it rather
# than silently discarding part of its gold answer.
for name, df in frames.items():
    for i, r in df.iterrows():
        if r.category != "Attribute_Characteristics" or r.sub_category != "Surface_or_secondary_change":
            continue
        if shape_configuration.search(" ".join([str(r.question), str(r.answer), " ".join(map(str, concepts(r.answer_concepts))), " ".join(options(r.question).values())])):
            quarantine_row(name, i, "configuration_mixed_into_surface_field", "Câu thuộc trường bề mặt/thứ phát nhưng còn dạng lưới/vân lưới; gold gộp hai trường nên cách ly.")

# Lesion Recognition: distinguish primary/secondary morphology and route appendage/pigment clues.
primary = re.compile(r"(?i)\b(?:dát|sẩn|mảng|nốt|u|mụn nước|bọng nước|mụn mủ|nang|sẩn phù|papule|nodule|plaque|vesicle|bulla|pustule|macule|patch|wheal|comedone|nhân trứng cá|nhân mụn|mụn đầu đen|mụn đầu trắng)\b")
secondary = re.compile(r"(?i)\b(?:vảy|vảy tiết|trợt|loét|nứt|sẹo|teo|lichen hóa|hoại tử|bong|tróc|mài|crust|erosion|ulcer|fissure|scar|necrosis)\b")
nail = re.compile(r"(?i)\b(?:móng|tách móng|dày móng|loạn dưỡng móng|ly móng|rãnh móng|bong móng)\b")
hair = re.compile(r"(?i)\b(?:tóc|lông|rụng tóc|thưa tóc|chấy|ký sinh trùng)\b")
pigment = re.compile(r"(?i)\b(?:tăng sắc tố|giảm sắc tố|dát nâu|đốm nâu|đốm trắng)\b")
redness = re.compile(r"(?i)\b(?:ban đỏ|hồng ban|đỏ hồng|đỏ)\b")
swelling = re.compile(r"(?i)\b(?:sưng nề|sưng phù|phù nề|phù|sưng môi|sưng ngón|sưng)\b")
nonvisual_tactile = re.compile(r"(?i)\b(?:mô mềm|mềm khi sờ|độ mềm)\b")
ambiguous = re.compile(r"(?i)\b(?:u mụn|sẩn sợi|nốt nhỏ|bọng nước nhỏ|bọng mủ|dưới da|dày sừng quang hóa|viêm quanh móng|viêm vách|viêm lợi|đốm|chấm|nốt sẩn|sẩn cục|sẩn dạng nốt|papulonodule|nốt mô hạt|khối lồi|nốt sẩn cuống|mảng|plaque|patch)\b")
for name, df in frames.items():
    for i, r in df.iterrows():
        if r.category != "Lesion_Recognition" or (str(r["index"]) == "0" and r.source_disease == "Acanthoma fissuratum"):
            continue
        opts = options(r.question) if r.type == "Multi_choice" else {}
        proposition = claim(r.question) if r.type == "Judgement" else " ".join(map(str, concepts(r.answer_concepts)))
        full = " ".join([r.question, r.answer, proposition, " ".join(opts.values())])
        if ambiguous.search(full):
            quarantine_row(name,i,"ambiguous_or_nonmorphologic_lesion_label","Gold/lựa chọn có thuật ngữ mơ hồ, chẩn đoán thay vì hình thái, hoặc phụ thuộc kích thước chưa xác minh.")
            continue
        # Wheal is a lesion morphology (sẩn phù); urticaria/mày đay is a disease label.
        if re.search(r"(?i)\bmày đay\b", full):
            oldq, olda = r.question, r.answer
            df.at[i,"question"] = replace_ci(r.question,"mày đay","sẩn phù")
            df.at[i,"answer"] = replace_ci(r.answer,"mày đay","sẩn phù")
            if r.type == "Multi_choice" and len(opts) == 4:
                set_mcq(df,i,{k:replace_ci(v,"mày đay","sẩn phù") for k,v in opts.items()})
            else:
                vals=[replace_ci(x,"mày đay","sẩn phù") for x in concepts(r.answer_concepts)]
                df.at[i,"answer_concepts"]=jl(vals)
                if r.type not in ("Judgement","Multi_choice"): df.at[i,"answer"]=format_vi(vals)
            mark_fix(name,df,i,"urticaria_name_used_as_morphology","Trong câu hỏi hình thái, thay tên bệnh mày đay bằng tổn thương dạng sẩn phù.",oldq,olda,"mày_day_to_wheal_morphology")
        terms = claim(df.at[i,"question"]) if r.type == "Judgement" else " ".join(map(str,concepts(df.at[i,"answer_concepts"])))
        p,s,n,h,pg = bool(primary.search(terms)), bool(secondary.search(terms)), bool(nail.search(terms)), bool(hair.search(terms)), bool(pigment.search(terms))
        if n and not (p or s or h):
            oldq, olda = r.question, r.answer
            df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Characteristics","Nail_Morphology"
            df.at[i,"question"] = f"Trong ảnh có dấu hiệu móng ‘{terms}’ không?" if r.type == "Judgement" else "Dấu hiệu hình thái móng được ghi nhận trong ảnh là gì?"
            mark_fix(name,df,i,"nail_finding_mislabeled_as_basic_lesion","Chuyển dấu hiệu móng sang nhóm đặc điểm móng.",oldq,olda,"nail_finding_reclassified")
        elif h and not (p or s or n):
            oldq, olda = r.question, r.answer
            df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Characteristics","Hair_Morphology"
            df.at[i,"question"] = f"Trong ảnh có dấu hiệu tóc/lông ‘{terms}’ không?" if r.type == "Judgement" else "Dấu hiệu tóc/lông được ghi nhận trong ảnh là gì?"
            mark_fix(name,df,i,"hair_finding_mislabeled_as_basic_lesion","Chuyển dấu hiệu tóc/lông sang nhóm đặc điểm tóc/lông.",oldq,olda,"hair_finding_reclassified")
        elif pg and not (p or s or n or h):
            oldq, olda = r.question, r.answer
            df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Color","Pigmentation"
            df.at[i,"question"] = f"Trong ảnh có đặc điểm sắc tố ‘{terms}’ không?" if r.type == "Judgement" else "Đặc điểm sắc tố của tổn thương là gì?"
            mark_fix(name,df,i,"pigment_mislabeled_as_basic_lesion","Chuyển dấu hiệu sắc tố thuần sang Attribute_Color.",oldq,olda,"pigment_reclassified")
        elif p and s:
            oldq, olda = r.question, r.answer
            df.at[i,"sub_category"]="Primary_and_Secondary_Morphology"
            if r.type == "Short_answer": df.at[i,"question"]="Trong ảnh, những tổn thương cơ bản và biến đổi thứ phát nào được quan sát thấy?"
            elif r.type == "Fill_in_blank": df.at[i,"question"]="Trong ảnh, các tổn thương cơ bản và biến đổi thứ phát là ____."
            mark_fix(name,df,i,"primary_and_secondary_morphology_mixed","Nêu rõ câu hỏi bao gồm cả tổn thương cơ bản và biến đổi thứ phát.",oldq,olda,"primary_secondary_scope_explicit")
        elif s:
            oldq, olda = r.question, r.answer
            df.at[i,"sub_category"]="Secondary_Change"
            if r.type == "Short_answer": df.at[i,"question"]="Trong ảnh, biến đổi thứ phát chính là gì?"
            elif r.type == "Fill_in_blank": df.at[i,"question"]="Trong ảnh, biến đổi thứ phát chính là ____."
            elif r.type == "Judgement": df.at[i,"question"] = f"Trong ảnh có {claim(oldq)} không?"
            elif r.type == "Multi_choice":
                header="Trong ảnh, biến đổi thứ phát nào được ghi nhận?\nChọn tất cả đáp án đúng."
                df.at[i,"question"]=header+"\n"+"\n".join(f"{k}. {v}" for k,v in options(oldq).items())
            mark_fix(name,df,i,"secondary_change_asked_as_primary","Đổi câu hỏi/sub_category để hỏi biến đổi thứ phát.",oldq,olda,"secondary_change_question_rephrased")
        elif p:
            df.at[i,"sub_category"]="Primary_Lesion_Type"

# Strict second pass: every retained Lesion_Recognition item must be a primary
# morphology question, an explicit primary+secondary question, or be routed to
# its true visual field. Unsupported labels are quarantined instead of inferred
# from the disease name.
for name, df in frames.items():
    for i in df.index:
        r = df.loc[i]
        if r.category != "Lesion_Recognition" or (str(r["index"]) == "0" and r.source_disease == "Acanthoma fissuratum"):
            continue
        oldq, olda = r.question, r.answer
        opts = options(oldq) if r.type == "Multi_choice" else {}
        labels = [claim(oldq)] if r.type == "Judgement" else [str(x) for x in concepts(r.answer_concepts)]
        label_text = " ".join(labels)
        option_text = " ".join(opts.values())
        if nonvisual_tactile.search(label_text) or (swelling.search(label_text) and primary.search(label_text)):
            quarantine_row(name, i, "nonvisual_or_mixed_field_lesion_gold", "Gold có dấu hiệu sờ khám hoặc trộn sưng/phù với hình thái cơ bản; không thể xác nhận/tách từ ảnh một cách đáng tin cậy.")
            continue
        if ambiguous.search(label_text) or (r.type == "Multi_choice" and ambiguous.search(option_text)):
            quarantine_row(name, i, "ambiguous_primary_morphology_term", "Còn thuật ngữ hình thái mơ hồ hoặc không chuẩn hóa được an toàn (ví dụ mảng patch/plaque, sẩn-cục hoặc nốt mô hạt).")
            continue
        p, s = bool(primary.search(label_text)), bool(secondary.search(label_text))
        n, h, pg = bool(nail.search(label_text)), bool(hair.search(label_text)), bool(pigment.search(label_text))
        if r.type == "Judgement":
            if not labels[0]:
                quarantine_row(name, i, "unparsed_lesion_judgement_claim", "Không trích xuất được mệnh đề hình thái từ câu Có/Không.")
                continue
            if p and s:
                df.at[i,"sub_category"]="Primary_and_Secondary_Morphology"
                df.at[i,"question"] = f"Trong ảnh có tổn thương cơ bản và biến đổi thứ phát như ‘{labels[0]}’ không?"
            elif s:
                df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Characteristics","Secondary_Change"
                df.at[i,"question"] = f"Trong ảnh có biến đổi thứ phát ‘{labels[0]}’ không?"
            elif p:
                df.at[i,"sub_category"]="Primary_Lesion_Type"
            elif redness.search(label_text) and not swelling.search(label_text):
                df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Color","Color"
                df.at[i,"question"]="Trong ảnh có vùng da đỏ/hồng ban không?"
            elif swelling.search(label_text) and not nonvisual_tactile.search(label_text):
                df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Characteristics","Swelling"
                df.at[i,"question"] = f"Trong ảnh có dấu hiệu sưng/phù ‘{labels[0]}’ không?"
            elif n or h or pg:
                quarantine_row(name, i, "appendage_or_pigment_claim_in_lesion_recognition", "Câu hỏi tổn thương cơ bản thực ra hỏi dấu hiệu móng/tóc/sắc tố; không thể đổi nhãn Có/Không một cách an toàn.")
                continue
            else:
                quarantine_row(name, i, "nonmorphologic_or_unresolved_lesion_judgement", "Mệnh đề không phải loại tổn thương cơ bản hoặc biến đổi thứ phát đã chuẩn hóa.")
                continue
            if df.at[i,"question"] != oldq or df.at[i,"category"] != r.category or df.at[i,"sub_category"] != r.sub_category:
                mark_fix(name,df,i,"lesion_judgement_scope_corrected","Đổi câu hỏi và trường dữ liệu cho khớp đúng loại hình thái được hỏi.",oldq,olda,"lesion_judgement_scope_corrected")
            continue

        if not labels or any(not (primary.search(x) or secondary.search(x)) for x in labels):
            if all(redness.search(x) and not (primary.search(x) or secondary.search(x)) and not swelling.search(x) for x in labels) and labels:
                if r.type == "Multi_choice" and any(not redness.search(x) for x in opts.values()):
                    quarantine_row(name, i, "mixed_color_and_morphology_mcq", "MCQ trộn lựa chọn màu sắc với hình thái tổn thương; không thể giữ một khung câu hỏi duy nhất.")
                    continue
                df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Color","Color"
                vals=["Đỏ" if x.strip().casefold() in {"ban đỏ","hồng ban"} else canon(x) for x in labels]
                if r.type == "Multi_choice":
                    op={k:("Đỏ" if v.strip().casefold() in {"ban đỏ","hồng ban"} else v) for k,v in opts.items()}
                    set_mcq(df,i,op)
                else:
                    df.at[i,"answer_concepts"]=jl(vals)
                    df.at[i,"answer"]=format_vi(vals)
                df.at[i,"question"]="Màu sắc/vùng đổi màu quan sát được trong ảnh là gì?"
                mark_fix(name,df,i,"redness_mislabeled_as_basic_lesion","Chuyển đáp án chỉ mô tả hồng ban sang trường màu sắc.",oldq,olda,"redness_reclassified_to_color")
                continue
            if labels and all(swelling.search(x) and not nonvisual_tactile.search(x) for x in labels):
                if r.type == "Multi_choice" and any(not swelling.search(x) for x in opts.values()):
                    quarantine_row(name, i, "mixed_swelling_and_morphology_mcq", "MCQ trộn dấu hiệu sưng/phù với lựa chọn hình thái; cách ly để tránh sai trường.")
                    continue
                df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Characteristics","Swelling"
                if r.type == "Multi_choice":
                    df.at[i,"question"]="Trong ảnh, dấu hiệu sưng/phù nào được ghi nhận?\nChọn tất cả đáp án đúng.\n"+"\n".join(f"{k}. {v}" for k,v in opts.items())
                else:
                    df.at[i,"question"]="Trong ảnh có dấu hiệu sưng/phù nào?" if r.type == "Short_answer" else "Trong ảnh, dấu hiệu sưng/phù là ____ .".replace(" ____", "____")
                mark_fix(name,df,i,"swelling_mislabeled_as_basic_lesion","Chuyển dấu hiệu sưng/phù thuần sang đặc điểm quan sát được.",oldq,olda,"swelling_reclassified_to_characteristics")
                continue
            quarantine_row(name, i, "nonmorphologic_or_unresolved_lesion_answer", "Đáp án không khớp loại tổn thương cơ bản hoặc biến đổi thứ phát đã chuẩn hóa; cách ly thay vì suy diễn từ tên bệnh.")
            continue

        option_p = bool(primary.search(option_text))
        option_s = bool(secondary.search(option_text))
        if r.type == "Multi_choice" and any(not (primary.search(v) or secondary.search(v)) for v in opts.values()):
            quarantine_row(name, i, "mcq_mixed_unmapped_lesion_options", "Một hoặc nhiều lựa chọn MCQ không phải thuật ngữ tổn thương cơ bản/biến đổi thứ phát chuẩn.")
            continue
        if p and s:
            df.at[i,"sub_category"]="Primary_and_Secondary_Morphology"
            header="Trong ảnh, những tổn thương cơ bản hoặc biến đổi thứ phát nào được quan sát?\nChọn tất cả đáp án đúng."
            if r.type == "Multi_choice":
                df.at[i,"question"]=header+"\n"+"\n".join(f"{k}. {v}" for k,v in opts.items())
            elif r.type == "Short_answer":
                df.at[i,"question"]="Trong ảnh, những tổn thương cơ bản và biến đổi thứ phát nào được quan sát thấy?"
            else:
                df.at[i,"question"]="Trong ảnh, các tổn thương cơ bản và biến đổi thứ phát là ____ .".replace(" ____", "____")
        elif s and not p:
            if r.type == "Multi_choice" and option_p:
                df.at[i,"sub_category"]="Primary_and_Secondary_Morphology"
                df.at[i,"question"]="Trong ảnh, những tổn thương cơ bản hoặc biến đổi thứ phát nào được quan sát?\nChọn tất cả đáp án đúng.\n"+"\n".join(f"{k}. {v}" for k,v in opts.items())
            else:
                df.at[i,"category"],df.at[i,"sub_category"]="Attribute_Characteristics","Secondary_Change"
                if r.type == "Multi_choice":
                    df.at[i,"question"]="Trong ảnh, biến đổi thứ phát nào được ghi nhận?\nChọn tất cả đáp án đúng.\n"+"\n".join(f"{k}. {v}" for k,v in opts.items())
                elif r.type == "Short_answer":
                    df.at[i,"question"]="Trong ảnh, biến đổi thứ phát chính là gì?"
                else:
                    df.at[i,"question"]="Trong ảnh, biến đổi thứ phát chính là ____ .".replace(" ____", "____")
        else:
            df.at[i,"sub_category"]="Primary_Lesion_Type"
            if r.type == "Multi_choice" and option_s:
                df.at[i,"sub_category"]="Primary_and_Secondary_Morphology"
                df.at[i,"question"]="Trong ảnh, những tổn thương cơ bản hoặc biến đổi thứ phát nào được quan sát?\nChọn tất cả đáp án đúng.\n"+"\n".join(f"{k}. {v}" for k,v in opts.items())
        if df.at[i,"question"] != oldq or df.at[i,"category"] != r.category or df.at[i,"sub_category"] != r.sub_category:
            mark_fix(name,df,i,"lesion_question_scope_corrected","Đồng bộ câu hỏi/sub_category với nhóm hình thái xuất hiện trong gold và lựa chọn.",oldq,olda,"lesion_question_scope_corrected")

# Final guardrails after all wording/ontology normalizations. This catches field
# leakage and MCQ defects created by canonicalization (for example mày đay -> sẩn phù).
for name, df in frames.items():
    for i, r in df.iterrows():
        if r.category in ("Attribute_Shape", "Attribute_Characteristics"):
            field_gold = " ".join([str(r.answer), " ".join(map(str, concepts(r.answer_concepts)))])
            if dist_re.search(distribution_search_text(field_gold)):
                quarantine_row(name, i, "distribution_remains_in_visual_attribute_field", "Sau tách tự động vẫn còn pattern phân bố trong đáp án hình thái/bề mặt; loại để tránh giữ sai trường.")
                continue
            if r.category == "Attribute_Shape" and re.search(r"(?i)\bmảng\b", " ".join([field_gold, str(r.question)])):
                quarantine_row(name, i, "ambiguous_patch_plaque_term_in_shape_field", "‘Mảng’ là hình thái tổn thương, không phải hình dạng/ranh giới; nguồn không phân biệt mảng phẳng (patch) với mảng gồ (plaque). Cách ly cả khi xuất hiện trong claim của câu Judgement hoặc option.")
                continue
            if r.category == "Attribute_Characteristics" and re.search(r"(?i)\bphân bố\b", field_gold):
                quarantine_row(name, i, "distribution_word_in_characteristics_field", "Đáp án trộn pattern phân bố vào trường đặc điểm bề mặt/thứ phát; không thể tách chính xác thành một câu QA.")
                continue
        if r.type == "Multi_choice":
            opts = options(r.question)
            ans = str(r.answer).strip().upper()
            valid_letters = bool(re.fullmatch(r"[A-D]+", ans)) and len(set(ans)) == len(ans) and ans == "".join(sorted(ans))
            valid_options = set(opts) == set("ABCD") and all(v.strip() for v in opts.values()) and len({v.strip().casefold() for v in opts.values()}) == 4
            if not valid_options:
                quarantine_row(name, i, "mcq_missing_or_duplicate_option", "MCQ sau chuẩn hóa không còn đủ bốn lựa chọn khác nhau A-D.")
                continue
            if not valid_letters or not set(ans).issubset(opts):
                quarantine_row(name, i, "mcq_invalid_answer_key", "Khóa đáp án MCQ không phải tập chữ cái A-D hợp lệ, duy nhất và theo thứ tự.")
                continue
            expected = [opts[c] for c in ans]
            actual = concepts(r.answer_concepts)
            if [str(x).strip().casefold() for x in actual] != [str(x).strip().casefold() for x in expected]:
                oldq, olda = r.question, r.answer
                df.at[i, "answer_concepts"] = jl(expected)
                mark_fix(name, df, i, "mcq_answer_concepts_realigned", "Đồng bộ answer_concepts với các lựa chọn được khóa đáp án chọn.", oldq, olda, "mcq_answer_concepts_realigned")

# Keep Test_1of3 and Test decisions synchronized because the former is sampled
# from the latter. Detect duplicates before writing so the same source row is
# never kept in one split and quarantined in the other.
for name, df in frames.items():
    kept = df[~df.source_index.astype(str).isin(quarantine[name])].copy()
    duplicate = kept.duplicated(["image_id", "question", "answer"], keep="first")
    for i in kept.index[duplicate]:
        quarantine_row(name, i, "duplicate_question_answer_same_image", "Câu hỏi và đáp án trùng trên cùng ảnh sau chuẩn hóa.")
shared_test_quarantine = set(quarantine["Test"]) | set(quarantine["Test_1of3"])
for name in ("Test_1of3", "Test"):
    df = frames[name]
    for i, r in df.iterrows():
        if str(r.source_index) in shared_test_quarantine and str(r.source_index) not in quarantine[name]:
            quarantine_row(name, i, "paired_test_split_quarantine", "Cùng source_index đã bị cách ly ở Test hoặc Test_1of3; đồng bộ quyết định giữa hai split.")

# Rebuild quality marker, exclude duplicate rows, and write new outputs.
for name, df in frames.items():
    df["quality_tier"]="rule_checked_visual_not_fully_adjudicated"
    kept=df[~df.source_index.astype(str).isin(quarantine[name])].copy()
    kept=df[~df.source_index.astype(str).isin(quarantine[name])].copy()
    kept.to_csv(OUT/FILES[name][1],sep="\t",index=False,encoding="utf-8")
    print(f"{name}: input={len(df)} quarantined={len(quarantine[name])} output={len(kept)} images={kept.image_id.nunique()}")

ledger=pd.DataFrame(issues)
ledger=ledger.drop_duplicates(subset=["dataset","source_index","issue_code","action"],keep="first") if len(ledger) else ledger
ledger.to_csv(OUT/"DermNet_VQA_issue_ledger_20260923.tsv",sep="\t",index=False,encoding="utf-8")
print("ledger rows",len(ledger))
if len(ledger): print(ledger.groupby(["dataset","action"]).size().to_string())
