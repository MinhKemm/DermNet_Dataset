PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi trả lời ngắn loại tổn thương
Input: {image_path}, {lesion}
Question: Tổn thương trong ảnh này là gì?
"""

def build_prompt(image_path, lesion):
    return PROMPT_TEMPLATE.format(image_path=image_path, lesion=lesion)

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    lesion = extracted_facts.get('lesion', '')
    if not lesion or "không rõ" in lesion.lower() or "không xác định" in lesion.lower():
        return None
        
    question = "Tổn thương trong ảnh này là gì?"
    
    return {
        "image_path": image_path,
        "category": "Lesion_Recognition",
        "type": "Short_answer",
        "question": question,
        "answer": lesion
    }
