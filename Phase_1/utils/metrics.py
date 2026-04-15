import re
import string

def flatten_json_values(data: dict) -> str:
    """Trích xuất toàn bộ giá trị từ JSON, bỏ qua keys và metadata."""
    values = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith('_') or key.lower() == 'error':
                continue
            values.append(flatten_json_values(value))
    elif isinstance(data, list):
        for item in data:
            values.append(flatten_json_values(item))
    elif isinstance(data, (str, int, float, bool)):
        values.append(str(data))
    return " ".join(values)

def preprocess_text(text: str) -> set:
    """Làm sạch văn bản chuyên sâu cho tiếng Việt và thuật ngữ y tế."""
    if not text: return set()
    
    # Chuyển lowercase và thay thế các dấu gạch ngang/xuyệt bằng khoảng trắng
    text = text.lower().replace("-", " ").replace("/", " ")
    
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize và loại bỏ các từ vô nghĩa cực ngắn (1 ký tự) 
    tokens = [t for t in text.split() if len(t) > 1]
    
    return set(tokens)

def calculate_jaccard(json1: dict, json2: dict) -> float:
    """Tính Jaccard Similarity giữa 2 JSON dựa trên tập hợp từ vựng."""
    if not json1 or "error" in json1 or not json2 or "error" in json2:
        return 0.0

    set1 = preprocess_text(flatten_json_values(json1))
    set2 = preprocess_text(flatten_json_values(json2))
    
    if not set1 and not set2: return 1.0
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return float(len(intersection) / len(union))

if __name__ == "__main__":
    # Test nhanh
    d1 = {"Colour": "Hồng ban đỏ", "Shape": ["Tròn", "Bờ rõ"]}
    d2 = {"Colour": "Màu đỏ hồng", "Shape": ["Hình tròn, ranh giới rõ"]}
    print(f"Jaccard Score: {calculate_jaccard(d1, d2):.4f}")