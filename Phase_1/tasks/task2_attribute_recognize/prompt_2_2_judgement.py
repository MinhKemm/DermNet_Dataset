import random

PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi phán đoán đặc điểm tổn thương
Input: {image_path}, {key_attribute}, {attribute_value}, {target_flag}
Question: {question}
"""

def build_prompt(image_path, key_attribute, attribute_value, target_flag, question):
    return PROMPT_TEMPLATE.format(
        image_path=image_path,
        key_attribute=key_attribute,
        attribute_value=attribute_value,
        target_flag=target_flag,
        question=question
    )

def get_taxonomy_pool(taxonomy, key_attribute):
    from .prompt_2_1_multi_choice import get_taxonomy_pool
    return get_taxonomy_pool(taxonomy, key_attribute)

def get_judgement_question(key_attribute, value):
    mapping = {
        'Color': f"Trong ảnh được cung cấp, màu sắc tổn thương có phải là {value} không?",
        'Shape': f"Trong ảnh được cung cấp, hình dạng tổn thương có phải là {value} không?",
        'Distribution': f"Trong ảnh được cung cấp, kiểu phân bố có phải là {value} không?",
        'Boundary': f"Trong ảnh được cung cấp, đường bờ có phải là {value} không?",
        'Size': f"Trong ảnh được cung cấp, kích thước có phải là {value} không?",
        'Quantity': f"Trong ảnh được cung cấp, số lượng có phải là {value} không?"
    }
    return mapping.get(key_attribute.capitalize(), f"Trong ảnh được cung cấp, đặc điểm này có phải là {value} không?")

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    key_attribute = kwargs.get('KEY_ATTRIBUTE', 'Color')
    attribute_value = extracted_facts.get(key_attribute.lower(), '') or extracted_facts.get(key_attribute, '')
    
    if not attribute_value or attribute_value.lower() == 'không rõ':
        return None
        
    target_flag = kwargs.get('target_flag', random.choice(["CÓ", "KHÔNG"]))
    
    if target_flag == "CÓ":
        query_value = attribute_value
        answer = "Có"
    else:
        pool = get_taxonomy_pool(taxonomy, key_attribute)
        all_values = [v for v in pool if v != attribute_value]
        if not all_values:
            all_values = ["Giá trị khác"]
        query_value = random.choice(all_values)
        answer = "Không"
        
    question = get_judgement_question(key_attribute, query_value)
    
    return {
        "image_path": image_path,
        "category": "Attribute_Recognition",
        "type": "Judgement",
        "question": question,
        "answer": answer
    }
