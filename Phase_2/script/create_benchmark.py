import json
import random
import os

def create_benchmark(input_json_path, output_json_path, sample_size=2000):
    # 1. Đọc dữ liệu gốc
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. Lấy ngẫu nhiên 2000 câu hỏi (nếu tổng data > 2000)
    if len(data) > sample_size:
        val_data = random.sample(data, sample_size)
    else:
        val_data = data
        print(f"Lưu ý: Dữ liệu gốc chỉ có {len(data)} mẫu, ít hơn {sample_size}.")

    # 3. Thêm trường Image_Path cho dễ dàng xử lý ở các bước sau
    for item in val_data:
        disease_en = item['Disease_EN']
        image_id = item['Image_ID']
        # Ánh xạ đường dẫn thực tế theo cấu trúc thư mục của bạn
        item['Image_Path'] = f"dermnet-output/images/{disease_en}/{image_id}.jpg"
        
        # Đảm bảo Model_Prediction trống
        item['Model_Prediction'] = None

    # 4. Lưu ra thư mục data/ (hoặc val/)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã tạo thành công {len(val_data)} mẫu tại {output_json_path}")

# Chạy thử
create_benchmark(
    input_json_path='/Users/binhminh/Desktop/DermNet_Dataset/result.json', 
    output_json_path='val/benchmark_val_2000.json'
)