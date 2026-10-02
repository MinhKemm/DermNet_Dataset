import json

PROMPT_TEMPLATE = """Bạn là Bác sĩ Chuyên khoa Da liễu đang xây dựng benchmark VQA y tế tiếng Việt.
Dựa trên Tên bệnh tiếng Anh: {disease_name_en}, kiến thức bệnh học dưới đây và dữ liệu Taxonomy:

Kiến thức bệnh học:
{disease_knowledge}

Taxonomy:
{taxonomy_text}

Nhiệm vụ của bạn:
1) Ánh xạ tên bệnh từ tiếng Anh sang tiếng Việt (disease_name_vi) theo chuẩn Taxonomy.
2) Xác định thuộc tính chẩn đoán chính (key_diagnostic_attribute), bắt buộc phải là một trong các giá trị: [Size, Color, Shape, Quantity, Distribution, Boundary].
3) Nêu lý do lâm sàng (clinical_rationale).

Trích xuất thành định dạng JSON:
{{
    "disease_name_en": "{disease_name_en}",
    "disease_name_vi": "Tên tiếng Việt",
    "key_diagnostic_attribute": "Thuộc tính (Size|Color|Shape|Quantity|Distribution|Boundary)",
    "clinical_rationale": "Lý do"
}}
"""

def build_prompt(disease_name_en: str, disease_knowledge: str, taxonomy_text: str) -> str:
    """Tạo prompt cho Phase 1 Anchor Analysis."""
    return PROMPT_TEMPLATE.format(
        disease_name_en=disease_name_en,
        disease_knowledge=disease_knowledge,
        taxonomy_text=taxonomy_text
    )

def parse_response(raw_response: str) -> dict:
    """Phân tích kết quả JSON trả về từ mô hình."""
    try:
        # Tìm block JSON nếu có
        start = raw_response.find('{')
        end = raw_response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = raw_response[start:end]
            return json.loads(json_str)
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {}
