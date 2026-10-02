import random

PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi phán đoán vị trí giải phẫu
Input: {image_path}, {location}, {target_flag}
Question: Trong ảnh được cung cấp, vị trí của tổn thương có phải là {query_location} không?
"""

def build_prompt(image_path, location, target_flag, query_location):
    return PROMPT_TEMPLATE.format(
        image_path=image_path,
        location=location,
        target_flag=target_flag,
        query_location=query_location
    )

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    location = extracted_facts.get('location', '')
    if location in ["Không xác định được trên ảnh", "Không đủ dữ liệu quan sát", ""]:
        return None
        
    target_flag = kwargs.get('target_flag', random.choice(["CÓ", "KHÔNG"]))
    
    if target_flag == "CÓ":
        query_location = location
        answer = "Có"
    else:
        all_locations = [loc['label'] for loc in taxonomy.get('locations', []) if loc['label'] != location]
        query_location = random.choice(all_locations)
        answer = "Không"
        
    question = f"Trong ảnh được cung cấp, vị trí của tổn thương có phải là {query_location} không?"
    
    return {
        "image_path": image_path,
        "category": "Location_Recognition",
        "type": "Judgement",
        "question": question,
        "answer": answer
    }
