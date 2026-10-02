import random

PROMPT_TEMPLATE = """Role: Chuyên gia da liễu tạo câu hỏi trắc nghiệm vị trí giải phẫu
Input: {image_path}, {location}, {taxonomy_locations}
Question: Vùng cơ thể nào xuất hiện trong bức ảnh này?
{options}
"""

def build_prompt(image_path, location, taxonomy_locations, options):
    return PROMPT_TEMPLATE.format(
        image_path=image_path,
        location=location,
        taxonomy_locations=taxonomy_locations,
        options=options
    )

def generate_vqa(image_path, extracted_facts, taxonomy, **kwargs):
    location = extracted_facts.get('location', '')
    if location in ["Không xác định được trên ảnh", "Không đủ dữ liệu quan sát", ""]:
        return None
        
    all_locations = [loc['label'] for loc in taxonomy.get('locations', []) if loc['label'] != location]
    distractors = random.sample(all_locations, 3)
    choices = distractors + [location]
    random.shuffle(choices)
    
    letters = ['A', 'B', 'C', 'D']
    options_str = "\n".join([f"{letters[i]}. {choices[i]}" for i in range(4)])
    answer_letter = letters[choices.index(location)]
    
    question = f"Vùng cơ thể nào xuất hiện trong bức ảnh này?\n{options_str}"
    
    return {
        "image_path": image_path,
        "category": "Location_Recognition",
        "type": "Multi_choice",
        "question": question,
        "answer": answer_letter
    }
