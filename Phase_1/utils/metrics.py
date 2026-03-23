# src/utils/metrics.py
import re

def tokenize(text: str) -> set:
    """Chuẩn hóa và tách từ (tokenization) một chuỗi văn bản."""
    if not isinstance(text, str):
        text = str(text)
    # Đưa về chữ thường, loại bỏ dấu câu, tách theo khoảng trắng
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())

def calculate_jaccard(text1: str, text2: str) -> float:
    """
    Tính Jaccard Index giữa 2 đoạn văn bản.
    Trả về giá trị từ 0.0 (không giống) đến 1.0 (giống hoàn toàn).
    """
    set1 = tokenize(text1)
    set2 = tokenize(text2)
    
    if not set1 and not set2:
        return 1.0 # Cả hai đều rỗng (vd: cùng là 'Not available')
        
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union)

def compute_json_similarity(json1: dict, json2: dict) -> dict:
    """Tính trung bình Jaccard Index cho toàn bộ các keys trong JSON extraction."""
    scores = {}
    keys = ["Category", "Lesion_Type", "Colour", "Shape", "Distribution", "Characteristics"]
    
    total_score = 0
    valid_keys = 0
    
    for key in keys:
        val1 = str(json1.get(key, ""))
        val2 = str(json2.get(key, ""))
        score = calculate_jaccard(val1, val2)
        scores[key] = score
        total_score += score
        valid_keys += 1
        
    avg_score = total_score / valid_keys if valid_keys > 0 else 0
    scores["average_jaccard"] = avg_score
    return scores