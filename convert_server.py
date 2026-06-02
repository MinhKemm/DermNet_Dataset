import pandas as pd
import os

# 1. Khai báo đường dẫn file tsv đầu vào và đầu ra
# (Đoạn này vẫn dùng đường dẫn tương đối từ vị trí chạy file để đọc được file TSV)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_2/DermNet_Test_Val_4000_fixed/DermNet_Test_mac.tsv"
output_file = os.path.join(BASE_DIR, "Phase_2", "DermNet_Test_Val_4000_fixed", "DermNet_Test_mac_relative.tsv")

# Chuỗi gốc cố định bắt đầu của bộ data theo yêu cầu của bạn
dataset_root = "DermNet_Dataset/dermnet-output/images/"

def convert_to_relative_path(win_path):
    if pd.isna(win_path):
        return win_path
    
    # Chuyển toàn bộ dấu \ của Windows thành / để chuẩn hóa định dạng Linux/Mac
    path_normalized = str(win_path).replace('\\', '/')
    
    # Tìm cụm '/images/' để lấy phần tên bệnh và tên ảnh phía sau
    if '/images/' in path_normalized:
        relative_part = path_normalized.split('/images/')[-1]
    else:
        # Dự phòng nếu dòng nào đó không chứa cụm '/images/' thì lấy 2 cấp cuối (tên_bệnh/tên_ảnh.jpg)
        parts = path_normalized.split('/')
        if len(parts) >= 2:
            relative_part = f"{parts[-2]}/{parts[-1]}"
        else:
            relative_part = parts[-1]
        
    # Nối thẳng để tạo ra đường dẫn bắt đầu từ DermNet_Dataset
    return dataset_root + relative_part

# 2. Đọc file TSV sử dụng pandas
if not os.path.exists(input_file):
    print(f"❌ Không tìm thấy file đầu vào tại: {input_file}")
else:
    print("Đang đọc file TSV...")
    df = pd.read_csv(input_file, sep='\t')

    # 3. Áp dụng hàm sửa đường dẫn cho cột 'image_path'
    print("Đang chuẩn hóa đường dẫn ảnh (chỉ lấy từ DermNet_Dataset/)...")
    df['image_path'] = df['image_path'].apply(convert_to_relative_path)

    # In thử dòng đầu tiên để bạn check kết quả hiển thị
    print(f"👉 Kết quả test dòng đầu: {df['image_path'].iloc[0]}")

    # 4. Lưu lại thành file TSV mới
    df.to_csv(output_file, sep='\t', index=False)
    print(f"✅ Hoàn thành! Đã lưu file sạch tại: {output_file}")