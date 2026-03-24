import re
import string

def flatten_json_values(data: dict) -> str:
    """
    Hàm đệ quy để trích xuất toàn bộ giá trị (values) từ một JSON (dict) lồng nhau.
    Chỉ lấy nội dung, bỏ qua các keys và các trường metadata nội bộ (như _metadata).
    """
    values = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            # Bỏ qua các key metadata kỹ thuật, không phải là đặc điểm bệnh lý
            if key.startswith('_') or key.lower() == 'error':
                continue
            values.append(flatten_json_values(value))
    elif isinstance(data, list):
        for item in data:
            values.append(flatten_json_values(item))
    elif isinstance(data, str):
        values.append(data)
    elif isinstance(data, (int, float, bool)): # Ép kiểu các giá trị nguyên thủy về string
        values.append(str(data))
        
    # Nối tất cả thành một chuỗi dài, cách nhau bởi khoảng trắng
    return " ".join(values)

def preprocess_text(text: str) -> set:
    """
    Làm sạch văn bản và chuyển thành tập hợp (set) các từ (tokens) duy nhất.
    """
    if not text:
        return set()
        
    # 1. Chuyển thành chữ thường
    text = text.lower()
    
    # 2. Loại bỏ dấu câu (Punctuation)
    # Ví dụ: "ban đỏ, hình tròn." -> "ban đỏ hình tròn"
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tách từ (Tokenize) theo khoảng trắng
    tokens = text.split()
    
    # (Tùy chọn) 4. Có thể thêm list stopwords tiếng Việt cơ bản ở đây nếu cần
    # stopwords = {"và", "hoặc", "của", "các", "có", "là", "bị", "ở", "trên"}
    # tokens = [t for t in tokens if t not in stopwords]
    
    return set(tokens)

def calculate_jaccard(json1: dict, json2: dict) -> float:
    """
    Tính điểm Jaccard Similarity giữa 2 JSON dict.
    Công thức: |A ∩ B| / |A ∪ B|
    """
    # Xử lý trường hợp 1 trong 2 dict bị lỗi rỗng hoặc chứa key "error"
    if not json1 or "error" in json1:
        return 0.0
    if not json2 or "error" in json2:
        return 0.0

    # 1. Trích xuất toàn bộ chữ từ JSON thành 1 string dài
    str1 = flatten_json_values(json1)
    str2 = flatten_json_values(json2)
    
    # 2. Làm sạch và chuyển thành tập hợp từ vựng
    set1 = preprocess_text(str1)
    set2 = preprocess_text(str2)
    
    # Xử lý trường hợp cả 2 model đều trả về JSON rỗng (không có đặc điểm nào)
    if not set1 and not set2:
        return 1.0 # Rỗng giống rỗng -> Tương đồng 100%
    if not set1 or not set2:
        return 0.0 # Một bên có, một bên không -> Khác biệt hoàn toàn
        
    # 3. Tính toán Jaccard
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    jaccard_score = len(intersection) / len(union)
    
    return float(jaccard_score)

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test case mô phỏng 2 VLM sinh ra kết quả hơi khác nhau về cách dùng từ
    vlm1_mock = {
        "Colour": "Đỏ hồng",
        "Shape": ["Hình vân gỗ", "Đồng tâm"],
        "Distribution": "Lan tỏa diện rộng ở vùng thân mình",
        "_metadata": {"model": "qwen"} # Hàm flatten sẽ tự động bỏ qua dòng này
    }
    
    vlm2_mock = {
        "Colour": "Màu đỏ, hơi hồng",
        "Shape": "Hình vân gỗ, dạng đồng tâm",
        "Distribution": "Lan tỏa ở vùng thân mình",
        "_metadata": {"model": "llava"}
    }
    
    score = calculate_jaccard(vlm1_mock, vlm2_mock)
    print("--- Test Jaccard Similarity ---")
    print(f"Jaccard Score: {score:.4f}")
    
    if score >= 0.85:
         print("=> Đánh giá: Độ đồng thuận CAO.")
    elif score >= 0.5:
         print("=> Đánh giá: Độ đồng thuận TRUNG BÌNH (Cần Judge phân xử kỹ).")
    else:
         print("=> Đánh giá: Độ đồng thuận THẤP (Nguy cơ Hallucination/Mâu thuẫn cao).")