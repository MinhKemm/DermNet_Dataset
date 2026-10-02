import json

PROMPT_TEMPLATE = """Bạn là một chuyên gia da liễu.
Dựa trên các đặc điểm lâm sàng, tổn thương được xác định là {lesion}.
Lý do thực tế: {lesion_reasoning}

Hãy tạo 3 lý do sai (distractors) hợp lý trong ngữ cảnh lâm sàng da liễu nhưng không đúng với hình ảnh này để làm câu hỏi trắc nghiệm.
Trả về kết quả dưới định dạng JSON:
{{
    "options": [
        "A. [Lý do đúng]",
        "B. [Lý do sai 1]",
        "C. [Lý do sai 2]",
        "D. [Lý do sai 3]"
    ],
    "answer": "A"
}}
"""

def build_prompt(lesion: str, lesion_reasoning: str) -> str:
    """Tạo prompt để LLM sinh các lựa chọn sai."""
    return PROMPT_TEMPLATE.format(lesion=lesion, lesion_reasoning=lesion_reasoning)

def generate_vqa(image_path: str, extracted_facts: dict) -> dict:
    """
    Tạo VQA cho câu hỏi trắc nghiệm suy luận tổn thương.
    Lưu ý: Để đơn giản hóa khi không gọi LLM trực tiếp ở đây,
    fallback_options sẽ được sử dụng. Trong hệ thống thực, bạn gọi LLM với build_prompt.
    """
    lesion = extracted_facts.get('lesion', '')
    lesion_reasoning = extracted_facts.get('lesion_reasoning', '')
    
    question = f"Tại sao tổn thương này lại được xác định là {lesion}?"
    
    # Fallback cho lựa chọn
    options = f"A. {lesion_reasoning}\nB. Do nhiễm trùng vi khuẩn lây lan nhanh.\nC. Do dị ứng với các tác nhân tiếp xúc ngoài da.\nD. Là biểu hiện của một khối u lành tính bẩm sinh."
    full_question = f"{question}\n{options}"
    
    return {
        "image_path": image_path,
        "category": "Lesion_Reasoning",
        "type": "Multi_choice",
        "question": full_question,
        "answer": "A"
    }
