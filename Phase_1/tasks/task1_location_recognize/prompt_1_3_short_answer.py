PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi trả lời ngắn vị trí giải phẫu
Input: {image_path}, {location}
Question: Vùng cơ thể nào xuất hiện trong bức ảnh này?
"""

def build_prompt(image_path, location):
    return PROMPT_TEMPLATE.format(image_path=image_path, location=location)

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    location = extracted_facts.get('location', '')
    if location in ["Không xác định được trên ảnh", "Không đủ dữ liệu quan sát", ""]:
        return None
        
    question = "Vùng cơ thể nào xuất hiện trong bức ảnh này?"
    
    return {
        "image_path": image_path,
        "category": "Location_Recognition",
        "type": "Short_answer",
        "question": question,
        "answer": location
    }
