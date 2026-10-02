import random

PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi trắc nghiệm loại tổn thương
Input: {image_path}, {lesion}, {taxonomy_lesions}
Question: Tổn thương trong ảnh này là gì?
{options}
"""

def build_prompt(image_path, lesion, taxonomy_lesions, options):
    return PROMPT_TEMPLATE.format(
        image_path=image_path,
        lesion=lesion,
        taxonomy_lesions=taxonomy_lesions,
        options=options
    )

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    lesion = extracted_facts.get('lesion', '')
    if not lesion or "không rõ" in lesion.lower() or "không xác định" in lesion.lower():
        return None
        
    all_lesions = [item['name_vi'] for item in taxonomy.get('lesion_types', []) if item['name_vi'] != lesion]
    
    if len(all_lesions) < 3:
        all_lesions += ["Lựa chọn A", "Lựa chọn B", "Lựa chọn C"]
        
    distractors = random.sample(all_lesions, 3)
    choices = distractors + [lesion]
    random.shuffle(choices)
    
    letters = ['A', 'B', 'C', 'D']
    options_str = "\n".join([f"{letters[i]}. {choices[i]}" for i in range(4)])
    answer_letter = letters[choices.index(lesion)]
    
    question = f"Tổn thương trong ảnh này là gì?\n{options_str}"
    
    return {
        "image_path": image_path,
        "category": "Lesion_Recognition",
        "type": "Multi_choice",
        "question": question,
        "answer": answer_letter
    }
