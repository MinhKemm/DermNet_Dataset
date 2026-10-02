from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class VQAItem:
    image_path: str
    category: str
    type: str
    question: str
    answer: str

def vqa_to_dict(item: VQAItem) -> Dict[str, Any]:
    """Chuyển đổi VQAItem thành dict tương thích với VLMEvalKit TSV"""
    return {
        "image_path": item.image_path,
        "category": item.category,
        "type": item.type,
        "question": item.question,
        "answer": item.answer
    }

def build_multi_choice_question(stem: str, options_dict: Dict[str, str]) -> str:
    """Định dạng câu hỏi trắc nghiệm với các lựa chọn A, B, C, D"""
    question_parts = [stem]
    for label, text in options_dict.items():
        question_parts.append(f"{label}. {text}")
    return "\n".join(question_parts)

def get_correct_label(options_dict: Dict[str, str], correct_value: str) -> str:
    """Trả về nhãn chữ cái (A/B/C/D) cho đáp án đúng"""
    for label, text in options_dict.items():
        if text == correct_value:
            return label
    return ""
