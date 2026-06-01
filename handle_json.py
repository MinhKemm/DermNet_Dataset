import os
import json

# Đường dẫn đến thư mục gốc chứa các folder bệnh (theo ảnh là final_canonical_vi)
root_dir = '/Users/binhminh/Desktop/DermNet_Dataset/final_canonical_vi'

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(subdir, file)
            
            # 1. Lấy tên file không bao gồm phần mở rộng (.json)
            # Ví dụ: "acanthoma-fissuratum-01.json" -> "acanthoma-fissuratum-01"
            image_name_val = os.path.splitext(file)[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 2. Thêm trường image_name vào trong TRICH_XUAT_JSON
                if "TRICH_XUAT_JSON" in data:
                    data["TRICH_XUAT_JSON"]["image_name"] = image_name_val
                    
                    # 3. Ghi đè lại vào file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    print(f"Updated: {file}")
                else:
                    print(f"Skipped (No TRICH_XUAT_JSON key): {file}")
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")