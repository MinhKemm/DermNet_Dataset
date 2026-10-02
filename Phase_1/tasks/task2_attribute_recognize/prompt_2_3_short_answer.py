PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi trả lời ngắn đặc điểm tổn thương
Input: {image_path}, {key_attribute}, {attribute_value}
Question: {question}
"""

def build_prompt(image_path, key_attribute, attribute_value, question):
    return PROMPT_TEMPLATE.format(
        image_path=image_path,
        key_attribute=key_attribute,
        attribute_value=attribute_value,
        question=question
    )

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    key_attribute = kwargs.get('KEY_ATTRIBUTE', 'Color')
    attribute_value = extracted_facts.get(key_attribute.lower(), '') or extracted_facts.get(key_attribute, '')
    
    if not attribute_value or attribute_value.lower() == 'không rõ':
        return None
        
    from .prompt_2_1_multi_choice import get_question_stem
    question = get_question_stem(key_attribute)
    
    return {
        "image_path": image_path,
        "category": "Attribute_Recognition",
        "type": "Short_answer",
        "question": question,
        "answer": attribute_value
    }
