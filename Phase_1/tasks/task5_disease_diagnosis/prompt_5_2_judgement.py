import random

def build_prompt(disease_name: str) -> str:
    return f"Dựa vào hình ảnh lâm sàng, bệnh nhân có phải mắc bệnh {disease_name} không?"

def generate_vqa(image_path: str, extracted_facts: dict, target_flag: bool = True, taxonomy_data: dict = None) -> dict:
    diagnose = extracted_facts.get('diagnose', '')
    
    if target_flag:
        disease_name = diagnose
        answer = "Có"
    else:
        distractors = ["Viêm da cơ địa", "Vảy nến", "Nấm da", "Zona thần kinh"]
        if taxonomy_data:
            all_disease_dicts = taxonomy_data.get('diseases_vietnamese', []) + taxonomy_data.get('diseases_international', [])
            all_diseases = [d['name_vi'] for d in all_disease_dicts if d.get('name_vi')]
            valid_distractors = [d for d in all_diseases if d.lower() != diagnose.lower()]
            if valid_distractors:
                disease_name = random.choice(valid_distractors)
            else:
                disease_name = random.choice(distractors)
        else:
            disease_name = random.choice(distractors)
            
        answer = "Không"
        
    return {
        "image_path": image_path,
        "category": "Diagnosis",
        "type": "Judgement",
        "question": build_prompt(disease_name),
        "answer": answer
    }
