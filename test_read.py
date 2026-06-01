import pickle
import pandas as pd

with open("/Users/binhminh/Desktop/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Vintern-1B-v2/T20260519-135037/01_DermNet_Val_4k_mac.pkl", "rb") as f:
    data = pickle.load(f)

# Nếu data là DataFrame
if isinstance(data, pd.DataFrame):
    print("Số lượng mẫu đã predict:", len(data))
    print(data.head(10))  # Hiển thị 5 dòng đầu tiên của DataFrame
# Nếu data là list/dict
else:
    print("Kiểu dữ liệu:", type(data))
    print("Một vài dữ liệu mẫu:", str(data)[:1000])