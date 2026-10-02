import random

def build_prompt(lesion: str, lesion_reasoning: str, target_flag: bool) -> str:
    """Xây dựng câu hỏi Yes/No."""
    reasoning = lesion_reasoning if target_flag else "có sự tập trung của nhiều tế bào sừng và tăng sản biểu bì sai lệch"
    return f"Dựa trên các đặc điểm hình thái quan sát được, nhận định tổn thương thuộc nhóm {lesion} vì {reasoning} có phù hợp không?"

def generate_vqa(image_path: str, extracted_facts: dict, target_flag: bool = True) -> dict:
    """
    Tạo VQA phán đoán (Judgement).
    Nếu target_flag = True -> CÓ.
    Nếu target_flag = False -> KHÔNG.
    """
    lesion = extracted_facts.get('lesion', '')
    lesion_reasoning = extracted_facts.get('lesion_reasoning', '')
    
    question = build_prompt(lesion, lesion_reasoning, target_flag)
    answer = "Có" if target_flag else "Không"
    
    return {
        "image_path": image_path,
        "category": "Lesion_Reasoning",
        "type": "Judgement",
        "question": question,
        "answer": answer
    }
