import json

PROMPT_TEMPLATE = """Bạn là Bác sĩ Da liễu trích xuất đặc điểm thị giác từ hình ảnh.
Tên bệnh tiếng Anh: {disease_name_en}
Tên bệnh tiếng Việt: {disease_name_vi}
Thuộc tính chẩn đoán chính (Key Attribute): {key_attribute}

Taxonomy Whitelist:
{taxonomy_text}

Dựa vào hình ảnh, hãy trích xuất các đặc điểm quan sát được (location, size, color, shape, quantity, distribution, boundary, lesion, lesion_reasoning, diagnose).
Yêu cầu bắt buộc:
- Mọi giá trị được trích xuất PHẢI NẰM TRONG danh sách Taxonomy Whitelist cho thuộc tính tương ứng.
- Nếu không thể xác định được thuộc tính nào đó trên ảnh, hãy điền: "Không xác định được trên ảnh".

Trả về JSON có cấu trúc như sau:
{{
    "extracted_facts": {{
        "location": "...",
        "size": "...",
        "color": "...",
        "shape": "...",
        "quantity": "...",
        "distribution": "...",
        "boundary": "...",
        "lesion": "...",
        "lesion_reasoning": "...",
        "diagnose": "{disease_name_vi}"
    }}
}}
"""

def build_prompt(disease_name_en: str, disease_name_vi: str, key_attribute: str, taxonomy_text: str) -> str:
    """Tạo prompt cho Phase 2 Constrained Fact Extraction."""
    return PROMPT_TEMPLATE.format(
        disease_name_en=disease_name_en,
        disease_name_vi=disease_name_vi,
        key_attribute=key_attribute,
        taxonomy_text=taxonomy_text
    )

def parse_response(raw_response: str) -> dict:
    """Phân tích JSON từ Phase 2."""
    try:
        start = raw_response.find('{')
        end = raw_response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = raw_response[start:end]
            return json.loads(json_str)
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"extracted_facts": {}}
