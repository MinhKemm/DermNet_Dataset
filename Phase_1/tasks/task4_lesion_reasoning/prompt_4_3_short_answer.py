def build_prompt(lesion: str) -> str:
    """Xây dựng câu hỏi ngắn."""
    return f"Tại sao tổn thương trong ảnh được xác định là {lesion}?"

def generate_vqa(image_path: str, extracted_facts: dict) -> dict:
    """Tạo VQA trả lời ngắn."""
    lesion = extracted_facts.get('lesion', '')
    lesion_reasoning = extracted_facts.get('lesion_reasoning', '')
    
    question = build_prompt(lesion)
    
    return {
        "image_path": image_path,
        "category": "Lesion_Reasoning",
        "type": "Short_answer",
        "question": question,
        "answer": lesion_reasoning
    }
