import pandas as pd

# 1. Đọc dữ liệu (Thay 'dataset.csv' bằng đường dẫn file thực tế của bạn)
# Nếu bạn lưu dạng file text phân tách bằng tab (TSV), hãy thêm sep='\t'
df = pd.read_csv("/Users/binhminh/Desktop/DermNet_Dataset/Phase_2/DermNet_Test_Val/DermNet_Test_mac.tsv", sep='\t')  # Hoặc pd.read_csv("dataset.csv") nếu là CSV thông thường

# 2. Kiểm tra điều kiện: image_path KHÔNG kết thúc bằng '.jpg'
# Dùng dấu ~ để lấy phủ định (NOT)
# na=False để bỏ qua/xử lý nếu có hàng nào bị khuyết thiếu dữ liệu (NaN)
not_jpg_mask = ~df["image_path"].str.endswith(".jpg", na=False)

# 3. Đếm số lượng index thỏa mãn
count_not_jpg = not_jpg_mask.sum()

# 4. In kết quả
print(f"Số lượng index có image_path không có đuôi .jpg là: {count_not_jpg}")

# --- BONUS: Nếu bạn muốn xem cụ thể các dòng đó như thế nào ---
if count_not_jpg > 0:
    print("\nDanh sách các dòng không có đuôi .jpg:")
    print(df[not_jpg_mask][["index", "image_path"]])