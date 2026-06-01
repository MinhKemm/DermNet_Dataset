import json
import os

def merge_for_judge(benchmark_path, prediction_path, output_path):
    # 1. Đọc file Gốc (thành dictionary để tra cứu nhanh)
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    
    benchmark_dict = {item['Question_ID']: item for item in benchmark_data}

    # 2. Đọc file Prediction của VLM
    with open(prediction_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    merged_data = []

    # 3. Gộp dữ liệu
    for pred in predictions:
        q_id = pred['Question_ID']
        if q_id in benchmark_dict:
            base_item = benchmark_dict[q_id]
            
            # Tạo object mới chỉ chứa các thông tin cần thiết cho Judge Model
            merged_item = {
                "Question_ID": q_id,
                "Task_Category": base_item['Task_Category'],
                "Question": base_item['Question'],
                "Ground_Truth": base_item['Ground_Truth'],
                "Model_Prediction": pred['Model_Prediction'],
                # Khởi tạo sẵn 2 trường này để GPT-5.5 điền vào
                "Judge_Score": None,
                "Judge_Explanation": None
            }
            merged_data.append(merged_item)

    # 4. Lưu ra file chờ chấm
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã gộp thành công {len(merged_data)} mẫu chờ chấm tại {output_path}")

# Gộp dữ liệu của gpt4o
merge_for_judge(
    benchmark_path="data/benchmark_val_2000.json",
    prediction_path="Ans/gpt4o/prediction.json",
    output_path="evaluation/to_be_judged_gpt4o.json"
)