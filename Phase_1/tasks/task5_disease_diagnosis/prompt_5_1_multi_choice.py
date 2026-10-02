import random

def build_prompt(options_text: str) -> str:
    return f"Trong hình ảnh này, chẩn đoán nào phù hợp nhất?\n{options_text}"

def generate_vqa(image_path: str, extracted_facts: dict, taxonomy_data: dict = None) -> dict:
    diagnose = extracted_facts.get('diagnose', '')
    
    distractors = ["Viêm da cơ địa", "Vảy nến", "Nấm da", "Mụn trứng cá", "Ung thư biểu mô tế bào đáy"]
    if taxonomy_data:
        all_disease_dicts = taxonomy_data.get('diseases_vietnamese', []) + taxonomy_data.get('diseases_international', [])
        all_diseases = [d['name_vi'] for d in all_disease_dicts if d.get('name_vi')]
        # Lọc bỏ bệnh đúng
        valid_distractors = [d for d in all_diseases if d.lower() != diagnose.lower()]
        if len(valid_distractors) >= 3:
            distractors = random.sample(valid_distractors, 3)
        else:
            distractors = random.sample(distractors, 3)
    else:
        distractors = random.sample(distractors, 3)
        
    choices = distractors + [diagnose]
    random.shuffle(choices)
    
    correct_idx = choices.index(diagnose)
    labels = ["A", "B", "C", "D"]
    
    options_text = "\n".join([f"{labels[i]}. {choices[i]}" for i in range(4)])
    answer = labels[correct_idx]
    
    return {
        "image_path": image_path,
        "category": "Diagnosis",
        "type": "Multi_choice",
        "question": build_prompt(options_text),
        "answer": answer
    }
