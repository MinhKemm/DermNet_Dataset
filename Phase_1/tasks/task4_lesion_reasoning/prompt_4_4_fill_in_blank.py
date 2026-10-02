import re
import random

def build_prompt(lesion: str, blank_sentence: str) -> str:
    """Xây dựng câu hỏi điền từ."""
    return f"Điền vào chỗ trống giải thích lý do tổn thương được xác định là {lesion}:\n{blank_sentence}"

def generate_vqa(image_path: str, extracted_facts: dict) -> dict:
    """Tạo VQA điền từ vào chỗ trống."""
    lesion = extracted_facts.get('lesion', '')
    reasoning = extracted_facts.get('lesion_reasoning', '')
    
    # Tìm các từ khóa lâm sàng
    keywords = re.findall(r'(kích thước|mô đặc|dịch|gồ|đỏ|nhám|sừng|vảy|bọng nước|viêm)', reasoning.lower())
    
    if keywords:
        key_term = random.choice(keywords)
        # Thay thế bằng ____
        # Dùng regex để thay thế case-insensitive nhưng lấy đúng từ gốc làm answer
        pattern = re.compile(re.escape(key_term), re.IGNORECASE)
        match = pattern.search(reasoning)
        if match:
            actual_term = match.group(0)
            blank_sentence = pattern.sub('____', reasoning, count=1)
            answer = actual_term
        else:
            blank_sentence = reasoning
            answer = ""
    else:
        # Fallback nếu không có từ khóa
        words = reasoning.split()
        if len(words) > 3:
            key_term = words[-2]
            blank_sentence = reasoning.replace(key_term, '____', 1)
            answer = key_term
        else:
            blank_sentence = reasoning
            answer = ""
            
    question = build_prompt(lesion, blank_sentence)
    
    return {
        "image_path": image_path,
        "category": "Lesion_Reasoning",
        "type": "Fill_in_blank",
        "question": question,
        "answer": answer
    }
