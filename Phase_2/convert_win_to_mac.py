import pandas as pd
import os

# 1. Khai báo đường dẫn file tsv đầu vào và đầu ra
input_file = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_2/DermNet_Test_Val_4000_fixed/DermNet_Val_4k.tsv" # Thay bằng tên file tsv hiện tại của bạn
output_file = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_2/DermNet_Test_Val_4000_fixed/DermNet_Val_4k_mac.tsv" # Tên file mới sau khi đã sửa xong đường dẫn

# Đường dẫn chuẩn trên Mac bạn muốn thay thế vào
mac_base_path = "/Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/images/"

def convert_to_mac_path(win_path):
    if pd.isna(win_path):
        return win_path
    
    # Chuyển toàn bộ dấu \ của Windows thành / của Mac
    path_normalized = str(win_path).replace('\\', '/')
    
    # Tìm cụm '/images/' để lấy phần tên bệnh và tên ảnh phía sau
    if '/images/' in path_normalized:
        # Lấy phần tử cuối cùng sau chữ /images/
        relative_part = path_normalized.split('/images/')[-1]
    else:
        # Trường hợp dự phòng nếu dòng nào đó không theo cấu trúc chuẩn
        relative_part = path_normalized.split('/')[-1]
        
    # Nối với đường dẫn gốc trên Mac
    return mac_base_path + relative_part

# 2. Đọc file TSV sử dụng pandas
print("Đang đọc file TSV...")
df = pd.read_csv(input_file, sep='\t')

# 3. Áp dụng hàm sửa đường dẫn cho cột 'image_path'
print("Đang chuẩn hóa đường dẫn ảnh sang định dạng macOS...")
df['image_path'] = df['image_path'].apply(convert_to_mac_path)

# 4. Lưu lại thành file TSV mới
df.to_csv(output_file, sep='\t', index=False)
print(f"Hoàn thành! Đã lưu file sạch tại: {output_file}")