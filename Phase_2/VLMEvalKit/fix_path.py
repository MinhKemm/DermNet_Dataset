import pandas as pd

input_file = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/deepseek_vl2_tiny/deepseek_vl2_tiny_DermNet_Val_4k.xlsx"  # hoặc .tsv, .csv
output_file = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/deepseek_vl2_tiny/deepseek_vl2_tiny_DermNet_Val_4k.xlsx"

# 1. Đọc file
if input_file.endswith(".xlsx"):
    df = pd.read_excel(input_file)
elif input_file.endswith(".tsv"):
    df = pd.read_csv(input_file, sep="\t")
else:
    df = pd.read_csv(input_file)

# 2. Chuẩn hóa tất cả dấu \ sang / (đề phòng đường dẫn Windows)
df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)

# 3. Lấy phần đuôi bắt đầu từ '/images/' rồi ghép với tiền tố mong muốn
# regex: '.*(?=/images/)' nghĩa là tìm và xóa sạch mọi thứ nằm trước '/images/'
df["image_path"] = df["image_path"].str.replace(
    r"^.*?/images/", "../../dermnet-output/images/", regex=True
)

# 4. Lưu lại
if output_file.endswith(".xlsx"):
    df.to_excel(output_file, index=False)
elif output_file.endswith(".tsv"):
    df.to_csv(output_file, sep="\t", index=False)
else:
    df.to_csv(output_file, index=False)

# In thử đầy đủ không bị cắt dấu '...'
for p in df["image_path"].head(3):
    print(p)