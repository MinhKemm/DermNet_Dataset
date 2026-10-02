import random

PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi phán đoán loại tổn thương
Input: {image_path}, {lesion}, {target_flag}
Question: Trong ảnh được cung cấp, tổn thương thực thể chính có phải là {query_lesion} không?
"""

def build_prompt(image_path, lesion, target_flag, query_lesion):
    return PROMPT_TEMPLATE.format(
        image_path=image_path,
        lesion=lesion,
        target_flag=target_flag,
        query_lesion=query_lesion
    )

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    lesion = extracted_facts.get('lesion', '')
    if not lesion or "không rõ" in lesion.lower() or "không xác định" in lesion.lower():
        return None
        
    target_flag = kwargs.get('target_flag', random.choice(["CÓ", "KHÔNG"]))
    
    if target_flag == "CÓ":
        query_lesion = lesion
        answer = "Có"
    else:
        all_lesions = [item['name_vi'] for item in taxonomy.get('lesion_types', []) if item['name_vi'] != lesion]
        query_lesion = random.choice(all_lesions) if all_lesions else "tổn thương khác"
        answer = "Không"
        
    question = f"Trong ảnh được cung cấp, tổn thương thực thể chính có phải là {query_lesion} không?"
    
    return {
        "image_path": image_path,
        "category": "Lesion_Recognition",
        "type": "Judgement",
        "question": question,
        "answer": answer
    }
