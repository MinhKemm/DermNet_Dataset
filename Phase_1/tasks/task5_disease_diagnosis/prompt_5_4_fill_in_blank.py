def build_prompt() -> str:
    return "Dựa vào hình ảnh lâm sàng, bệnh nhân được chẩn đoán mắc bệnh ____."

def generate_vqa(image_path: str, extracted_facts: dict) -> dict:
    diagnose = extracted_facts.get('diagnose', '')
    
    return {
        "image_path": image_path,
        "category": "Diagnosis",
        "type": "Fill_in_blank",
        "question": build_prompt(),
        "answer": diagnose
    }
