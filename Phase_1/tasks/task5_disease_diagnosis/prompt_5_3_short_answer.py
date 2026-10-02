def build_prompt() -> str:
    return "Có thể chẩn đoán bệnh gì dựa trên hình ảnh này?"

def generate_vqa(image_path: str, extracted_facts: dict) -> dict:
    diagnose = extracted_facts.get('diagnose', '')
    
    return {
        "image_path": image_path,
        "category": "Diagnosis",
        "type": "Short_answer",
        "question": build_prompt(),
        "answer": diagnose
    }
