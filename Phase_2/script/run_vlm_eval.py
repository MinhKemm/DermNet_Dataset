import json
import os

def run_inference(model_name, benchmark_path):
    # Đọc tập val
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    results = []
    
    print(f"Bắt đầu chạy inference cho model: {model_name}...")
    for item in val_data:
        question_id = item['Question_ID']
        question = item['Question']
        image_path = item['Image_Path']
        
        # -----------------------------------------------------
        # TODO: Thay thế đoạn này bằng code gọi Model/API thực tế
        # Ví dụ: response = model.generate(image_path, question)
        # -----------------------------------------------------
        mock_response = f"Đây là câu trả lời giả lập từ {model_name}" 
        
        # Chỉ lưu những thông tin tối giản nhất
        results.append({
            "Question_ID": question_id,
            "Model_Prediction": mock_response
        })

    # Lưu kết quả vào thư mục Ans/model_name/
    output_dir = f"Ans/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prediction.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Đã lưu kết quả của {model_name} tại {output_path}")

# Chạy thử với mô hình GPT-4o
run_inference(model_name="gpt4o", benchmark_path="data/benchmark_val_2000.json")