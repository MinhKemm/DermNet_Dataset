import pandas as pd
import glob

def clean_image_path(df):
    if 'image_path' in df.columns:
        # Chuẩn hóa dấu gạch chéo sang chuẩn Linux/Mac
        df['image_path'] = df['image_path'].astype(str).str.replace('\\', '/', regex=False)
        # Rút gọn 2 lần folder lồng nhau về 1 lần
        df['image_path'] = df['image_path'].str.replace('dermnet-output/dermnet-output/', 'dermnet-output/', regex=False)
    return df

# 1. Bỏ chữ 'DermNet_Dataset/' ở đầu vì đang đứng sẵn trong thư mục này
tsv_files = glob.glob("Phase_2/VLMEvalKit/LMUData/*.tsv")
for tsv_path in tsv_files:
    df_tsv = pd.read_csv(tsv_path, sep='\t')
    df_tsv = clean_image_path(df_tsv)
    df_tsv.to_csv(tsv_path, sep='\t', index=False)
    print(f"Đã cập nhật TSV: {tsv_path}")

# 2. Cập nhật các file Excel trong outputs
excel_files = glob.glob("Phase_2/VLMEvalKit/outputs/**/*.xlsx", recursive=True)
for excel_path in excel_files:
    df_excel = pd.read_excel(excel_path)
    df_excel = clean_image_path(df_excel)
    df_excel.to_excel(excel_path, index=False)
    print(f"Đã cập nhật Excel: {excel_path}")

print("Hoàn tất! Kiểm tra xem trên màn hình terminal có in ra các dòng 'Đã cập nhật...' không.")