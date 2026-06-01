import json
import pandas as pd
import os
import csv
import base64

def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def convert_json_to_tsv_with_base64(json_path, tsv_output_path):
    # Đường dẫn gốc dataset
    base_dir = "/Users/binhminh/Desktop/DermNet_Dataset"

    # Đọc JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []

    for item in data:
        # Tạo đường dẫn tuyệt đối
        img_path = os.path.abspath(
            os.path.join(base_dir, item['Image_Path'])
        )

        # Nếu file ảnh tồn tại -> convert sang base64
        if os.path.exists(img_path):
            img_base64 = image_to_base64(img_path)
        else:
            print(f"[WARNING] Không tìm thấy ảnh: {img_path}")
            img_base64 = ""

        raw_question = item['Question']
        q_type = item['Question_Type'] # Lấy type từ JSON (Multi_choice, Short_answer, ...)

        # =========================
        # Prompt tối ưu cho SmolVLM
        # =========================

        if q_type == "Multi_choice":

            refined_question = f"""
        Answer using ONLY the correct option letter.
        Do not explain.
        Trả lời chỉ bằng chữ cái đáp án.

        Question:
        {raw_question}

        Final Answer:
        """.strip()

        elif q_type == "Short_answer":

            refined_question = f"""
        Answer in Vietnamese.
        Use ONLY one short medical phrase.
        Maximum 3 words.
        Do not explain.

        Question:
        {raw_question}

        Final Answer:
        """.strip()

        elif q_type == "Fill_in_blank":

            refined_question = f"""
        Fill in the blank.
        Answer in Vietnamese.
        Use ONLY the missing word or phrase.
        Maximum 3 words.
        Do not explain.

        Question:
        {raw_question}

        Final Answer:
        """.strip()

        elif q_type == "Judgement":

            refined_question = f"""
        Answer using ONLY:
        - Đúng
        or
        - Sai

        Do not explain.

        Question:
        {raw_question}

        Final Answer:
        """.strip()

        else:

            refined_question = f"""
        Answer briefly in Vietnamese.
        Do not explain.

        Question:
        {raw_question}

        Final Answer:
        """.strip()

        processed_data.append({
            "index": item['Question_ID'],
            "image": img_base64,
            "question": refined_question,
            "answer": item['Ground_Truth'],
            "category": item['Task_Category'],
            "type": item['Question_Type']
        })

    # Tạo DataFrame
    df = pd.DataFrame(processed_data)

    # Xuất TSV
    df.to_csv(
        tsv_output_path,
        sep='\t',
        index=False,
        quoting=csv.QUOTE_ALL,  # Bao quanh mọi field bằng dấu ngoặc kép để tránh lỗi định dạng
        encoding='utf-8'
    )

    print(f"Đã tạo file TSV chứa base64 tại: {tsv_output_path}")

# =========================
# CHẠY
# =========================

json_in = "/Users/binhminh/Desktop/DermNet_Dataset/val/benchmark_val_2000.json"
tsv_out = "/Users/binhminh/Desktop/DermNet_Dataset/LMUData/DermNet_VQA_Bench.tsv"

convert_json_to_tsv_with_base64(json_in, tsv_out)