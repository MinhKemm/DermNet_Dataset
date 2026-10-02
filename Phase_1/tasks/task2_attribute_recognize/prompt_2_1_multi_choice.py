import random

PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi trắc nghiệm đặc điểm tổn thương
Input: {image_path}, {key_attribute}, {attribute_value}
Question: {question_stem}
{options}
"""

def build_prompt(image_path, key_attribute, attribute_value, question_stem, options):
    return PROMPT_TEMPLATE.format(
        image_path=image_path,
        key_attribute=key_attribute,
        attribute_value=attribute_value,
        question_stem=question_stem,
        options=options
    )

def get_taxonomy_pool(taxonomy, key_attribute):
    attr = key_attribute.lower()
    if attr == 'color':
        # Every alternate item is Vietnamese translation? No, wait. 
        # For simplicity, we just parse the correct list or use flat_attribute_values
        for text in taxonomy.get('flat_attribute_values', []):
            if 'Color' in text:
                return [x.strip() for x in taxonomy['flat_attribute_values'][taxonomy['flat_attribute_values'].index(text)+1].split(',')]
    elif attr == 'shape':
        for text in taxonomy.get('flat_attribute_values', []):
            if 'Shape' in text:
                return [x.strip() for x in taxonomy['flat_attribute_values'][taxonomy['flat_attribute_values'].index(text)+1].split(',')]
    elif attr == 'distribution':
        for text in taxonomy.get('flat_attribute_values', []):
            if 'Distribution' in text:
                return [x.strip() for x in taxonomy['flat_attribute_values'][taxonomy['flat_attribute_values'].index(text)+1].split(',')]
    elif attr == 'boundary':
        for text in taxonomy.get('flat_attribute_values', []):
            if 'Boundary' in text:
                return [x.strip() for x in taxonomy['flat_attribute_values'][taxonomy['flat_attribute_values'].index(text)+1].split(',')]
    elif attr == 'size':
        for text in taxonomy.get('flat_attribute_values', []):
            if 'Size' in text:
                return [x.strip() for x in taxonomy['flat_attribute_values'][taxonomy['flat_attribute_values'].index(text)+1].split(',')]
    elif attr == 'quantity':
        for text in taxonomy.get('flat_attribute_values', []):
            if 'Quantity' in text:
                return [x.strip() for x in taxonomy['flat_attribute_values'][taxonomy['flat_attribute_values'].index(text)+1].split(',')]
    return []

def get_question_stem(key_attribute):
    mapping = {
        'Color': "Màu sắc nào được ghi nhận ở tổn thương trong ảnh?",
        'Shape': "Đặc điểm hình dạng nào phù hợp với tổn thương trong ảnh?",
        'Distribution': "Kiểu phân bố nào được ghi nhận ở tổn thương?",
        'Boundary': "Đặc điểm đường bờ nào phù hợp với tổn thương trong ảnh?",
        'Size': "Kích thước nào phù hợp với tổn thương trong ảnh?",
        'Quantity': "Số lượng tổn thương trong ảnh được mô tả là gì?"
    }
    return mapping.get(key_attribute.capitalize(), "Đặc điểm nào phù hợp với tổn thương trong ảnh?")

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    key_attribute = kwargs.get('KEY_ATTRIBUTE', 'Color')
    # extracted_facts dùng key lowercase (color, shape, ...) nên cần normalize
    attribute_value = extracted_facts.get(key_attribute.lower(), '') or extracted_facts.get(key_attribute, '')
    
    if not attribute_value or attribute_value.lower() == 'không rõ':
        return None
        
    pool = get_taxonomy_pool(taxonomy, key_attribute)
    all_values = [v for v in pool if v != attribute_value]
    
    if len(all_values) < 3:
        # Fallback if pool doesn't have enough elements
        all_values += ["Lựa chọn A", "Lựa chọn B", "Lựa chọn C"]
        
    distractors = random.sample(all_values, 3)
    choices = distractors + [attribute_value]
    random.shuffle(choices)
    
    letters = ['A', 'B', 'C', 'D']
    options_str = "\n".join([f"{letters[i]}. {choices[i]}" for i in range(4)])
    answer_letter = letters[choices.index(attribute_value)]
    
    question_stem = get_question_stem(key_attribute)
    question = f"{question_stem}\n{options_str}"
    
    return {
        "image_path": image_path,
        "category": "Attribute_Recognition",
        "type": "Multi_choice",
        "question": question,
        "answer": answer_letter
    }
