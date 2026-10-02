PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi điền vào chỗ trống loại tổn thương
Input: {image_path}, {lesion}
Question: Dựa vào hình ảnh trên, loại tổn thương thực thể chính quan sát được là ____.
"""

def build_prompt(image_path, lesion):
    return PROMPT_TEMPLATE.format(image_path=image_path, lesion=lesion)

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    lesion = extracted_facts.get('lesion', '')
    if not lesion or "không rõ" in lesion.lower() or "không xác định" in lesion.lower():
        return None
        
    question = "Dựa vào hình ảnh trên, loại tổn thương thực thể chính quan sát được là ____."
    
    return {
        "image_path": image_path,
        "category": "Lesion_Recognition",
        "type": "Fill_in_blank",
        "question": question,
        "answer": lesion
    }
