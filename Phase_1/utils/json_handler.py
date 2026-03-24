import json
import os

def save_json(data: dict, filepath: str):
    """
    Lưu Dictionary Python thành file JSON đẹp (Pretty Print).
    Tự động tạo thư mục cha nếu chưa tồn tại.
    """
    # Lấy đường dẫn thư mục chứa file
    directory = os.path.dirname(filepath)
    
    # Tạo thư mục nếu nó chưa có (Ví dụ: test_demo/intermediate/)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # indent=4 giúp dễ nhìn trên VS Code
            # ensure_ascii=False giữ nguyên dấu tiếng Việt
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[+] Đã lưu thành công: {filepath}")
    except Exception as e:
        print(f"[-] [LỖI LƯU FILE] Không thể lưu {filepath}: {e}")