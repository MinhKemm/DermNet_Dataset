PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi điền vào chỗ trống đặc điểm tổn thương
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

def get_fill_blank_question(key_attribute):
    mapping = {
        'Color': "Dựa vào hình ảnh trên, tổn thương có màu sắc là ____.",
        'Shape': "Quan sát bức ảnh, mô tả hình thái phù hợp cho tổn thương là ____.",
        'Distribution': "Dựa vào hình ảnh trên, kiểu phân bố của tổn thương là ____.",
        'Boundary': "Quan sát bức ảnh, đặc điểm đường bờ phù hợp cho tổn thương là ____.",
        'Size': "Dựa vào hình ảnh trên, kích thước của tổn thương là ____.",
        'Quantity': "Quan sát bức ảnh, số lượng tổn thương là ____."
    }
    return mapping.get(key_attribute.capitalize(), "Dựa vào hình ảnh, đặc điểm tổn thương là ____.")

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    key_attribute = kwargs.get('KEY_ATTRIBUTE', 'Color')
    attribute_value = extracted_facts.get(key_attribute.lower(), '') or extracted_facts.get(key_attribute, '')
    
    if not attribute_value or attribute_value.lower() == 'không rõ':
        return None
        
    question = get_fill_blank_question(key_attribute)
    
    return {
        "image_path": image_path,
        "category": "Attribute_Recognition",
        "type": "Fill_in_blank",
        "question": question,
        "answer": attribute_value
    }
