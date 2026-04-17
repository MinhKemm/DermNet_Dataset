"""
config_loader.py — Load & parse YAML config cho Phase 1 pipeline
"""

import os
import re
import yaml


# ═══════════════════════════════════════════════════════════════
#  Path helpers
# ═══════════════════════════════════════════════════════════════
def get_project_root() -> str:
    """Ra thư mục gốc của project."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_file_dir, "../../"))


def load_yaml_config(file_path: str) -> dict:
    project_root = get_project_root()
    abs_path = file_path if os.path.isabs(file_path) else os.path.join(project_root, file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"[LỖI CONFIG] Không tìm thấy: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


# ═══════════════════════════════════════════════════════════════
#  Parse Phase 1 output (5 dòng key → value)
# ═══════════════════════════════════════════════════════════════
def parse_phase1_output(raw_text: str) -> dict:
    """
    Parse output Phase 1 từ định dạng:

        Loại tổn thương: Mảng hồng ban, Đỏ hồng
        Màu sắc: Đỏ nhạt, Trắng xám
        Hình dạng: Bất quy tắc, Ranh giới rõ
        Vị trí & sắp xếp: Thân mình, Khu trú
        Đặc điểm phụ: Bong vảy, Mụn mủ nhỏ

    Trả về dict:
        {
          "Lesion_Type":  ["Mảng hồng ban", "Đỏ hồng"],
          "Colour":       ["Đỏ nhạt", "Trắng xám"],
          "Shape":        ["Bất quy tắc", "Ranh giới rõ"],
          "Distribution": ["Thân mình", "Khu trú"],
          "Characteristics": ["Bong vảy", "Mụn mủ nhỏ"],
        }
    """
    field_map = {
        "loại tổn thương":  "Lesion_Type",
        "màu sắc":          "Colour",
        "hình dạng":        "Shape",
        "vị trí":           "Distribution",
        "đặc điểm phụ":     "Characteristics",
    }

    result = {v: [] for v in field_map.values()}
    lines = raw_text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Tìm key: value (hỗ trợ format "Key: value" hoặc "Key value")
        # Dùng regex để bắt "Loại tổn thương:" hoặc "Loại tổn thương"
        matched_key = None
        for label, json_key in fieldMap.items():
            if line.lower().startswith(label):
                matched_key = json_key
                value_str = line[len(label):].lstrip(" :\t")
                break

        if not matched_key:
            # Fallback: đoán key từ thứ tự dòng (0-4 → 5 trường)
            pass

        # Tách keywords bằng dấu phẩy
        keywords = [k.strip() for k in value_str.split(",") if k.strip()]
        result[matched_key].extend(keywords)

    return result


# ═══════════════════════════════════════════════════════════════
#  Prompt builder helpers
# ═══════════════════════════════════════════════════════════════
def build_phase1_user_prompt(p1_config: dict) -> str:
    """
    Ghép user_template + few_shot_examples cho Phase 1.
    Few-shot examples được ghép trực tiếp vào template.
    """
    template = p1_config.get("user_template", "")

    examples = p1_config.get("few_shot_examples", [])
    if not examples:
        return template

    section = "\n\n--- VÍ DỤ MINH HỌA ---\n"
    for ex_item in examples:
        for _, ex_val in ex_item.items():
            img_name = os.path.basename(ex_val.get("image_name", ""))
            output   = ex_val.get("expected_output", "").strip()
            section += f"[{img_name}]\n{output}\n\n"

    section += "--> BÂY GIỜ LÀ LƯỢT CỦA BẠN:\n"
    return template + section


def build_phase2_user_prompt(p2_config: dict,
                             phase1_output: str,
                             disease_name: str,
                             disease_knowledge: str) -> str:
    """
    Ghép user_template với giá trị thực + few_shot cho Phase 2.
    """
    template = p2_config.get("user_template", "")

    # Format template với giá trị
    formatted = template.format(
        phase1_qa_output  = phase1_output,
        disease_name      = disease_name,
        disease_knowledge = disease_knowledge or "(Không có kiến thức bệnh)",
    )

    # Thêm few-shot examples
    examples = p2_config.get("few_shot_examples", [])
    if examples:
        section = "\n\n--- VÍ DỤ MINH HỌA ---\n"
        for ex_item in examples:
            for _, ex_val in ex_item.items():
                d_name = ex_val.get("disease_name", "")
                p1_out = ex_val.get("phase1_qa_output", "").strip()
                exp_js = ex_val.get("expected_json", "").strip()
                section += f"[{d_name}]\nQA Đầu vào:\n{p1_out}\n\nJSON Đầu ra:\n{exp_js}\n\n"

        section += f"--> TRÍCH XUẤT JSON CHO BỆNH: {disease_name}\n"
        formatted += section

    return formatted


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════
def get_settings() -> dict:
    return load_yaml_config("Phase_1/config/settings.yaml")


def get_prompts() -> dict:
    return load_yaml_config("Phase_1/config/prompts.yaml")


# ═══════════════════════════════════════════════════════════════
#  Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Project Root: {get_project_root()}")

    try:
        settings = get_settings()
        print("✅ settings.yaml loaded")

        prompts = get_prompts()
        print("✅ prompts.yaml loaded")

        # Test parse Phase 1
        sample = """
Loại tổn thương: Mảng hồng ban, Sẩn đỏ
Màu sắc: Đỏ tươi, Trắng bạc
Hình dạng: Bất quy tắc, Ranh giới rõ
Vị trí & sắp xếp: Thân mình, Khu trú
Đặc điểm phụ: Bong vảy, Vảy nến
        """
        parsed = parse_phase1_output(sample)
        print("\nParsed Phase 1 output:")
        for k, v in parsed.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()