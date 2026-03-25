import json
import os

def save_json(data: dict, filepath: str):
    """
    Lưu Dictionary Python thành file JSON đẹp (Pretty Print).
    Tự động tạo thư mục cha nếu chưa tồn tại.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[-] [LỖI LƯU FILE] Không thể lưu {filepath}: {e}")

def save_json_ordered(data: dict, filepath: str):
    """
    Lưu JSON với thứ tự keys được ép cứng theo đúng luồng quan sát.
    Thứ tự chuẩn: Category -> Vị trí -> Loại tổn thương -> Màu sắc -> Hình dạng -> Đặc điểm
    """
    ordered_keys = [
        "Category", 
        "Distribution",    # 1. Vị trí
        "Lesion_Type",     # 2. Loại tổn thương
        "Colour",          # 3. Màu sắc
        "Shape",           # 4. Hình dạng
        "Characteristics", # 5. Đặc điểm nhận dạng phụ
        "_metadata", 
        "error"
    ]
    
    # Lọc và sắp xếp lại data theo đúng thứ tự mảng ordered_keys
    ordered_data = {k: data[k] for k in ordered_keys if k in data}
    
    # Nhặt nốt các key khác (nếu model có lỡ sinh thừa) ném vào cuối JSON
    for k, v in data.items():
        if k not in ordered_keys:
            ordered_data[k] = v
            
    # Tái sử dụng logic lưu file an toàn từ hàm save_json
    save_json(ordered_data, filepath)
    print(f"[+] Đã lưu thành công JSON (Ordered): {filepath}")