#!/bin/bash

# ==============================================================================
# Script Name: run_phase2.sh
# Description: Script đánh giá các VLM 
# Usage:
#   Full (Chạy cả bộ data):  bash run_phase2.sh full  <model_name> <test|val>
#   Patch (Chỉ sửa Lesion Reasoning): bash run_phase2.sh patch <model_name> <test|val> <excel_path>
# ==============================================================================

# cd DermNet_Dataset/Phase_2/VLMEvalKit || exit 1
pip install -r requirements.txt
pip uninstall -y torchaudio && pip install vllm

# Lấy đường dẫn tuyệt đối của thư mục hiện tại (do đã cd vào VLMEvalKit)
CURRENT_DIR="$PWD"

# Ép tuyệt đối biến LMUData để VLMEvalKit không bị nhảy về thư mục Home (~/LMUData)
export LMUData="$CURRENT_DIR/LMUData"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export HF_TOKEN="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"

MODE=$1
MODEL_NAME=$2
SPLIT=$3
EXCEL_PATH=$4

if [ -z "$MODE" ] || [ -z "$MODEL_NAME" ] || [ -z "$SPLIT" ]; then
    echo "Usage: bash run_phase2.sh <mode: full|patch> <model_name> <split: test|val> [excel_path]"
    exit 1
fi

# --- Xác định dataset dựa trên split ---
if [ "$SPLIT" == "test" ]; then
    BASE_DATASET="DermNet_Test_mac_relative"
    MINI_DATASET="DermNet_Test_Reasoning_Fix"
elif [ "$SPLIT" == "val" ]; then
    BASE_DATASET="DermNet_Val_4k-2_mac_relative"
    MINI_DATASET="DermNet_Val_4k-2_Reasoning_Fix"
else
    echo "Error: Invalid split '$SPLIT'. Must be 'test' or 'val'."
    exit 1
fi

BASE_TSV="LMUData/${BASE_DATASET}.tsv"
MINI_TSV="LMUData/${MINI_DATASET}.tsv"

# ==============================================================================
# LUỒNG 1: FULL RUN
# ==============================================================================
if [ "$MODE" == "full" ]; then
    echo ">>> Running FULL mode on $BASE_DATASET with model $MODEL_NAME"
    python run.py --data $BASE_DATASET --model "$MODEL_NAME" --mode infer --verbose --reuse --use-vllm

# ==============================================================================
# LUỒNG 2: PATCH RUN
# ==============================================================================
elif [ "$MODE" == "patch" ]; then
    if [ -z "$EXCEL_PATH" ]; then
        echo "Error: Excel path is required for patch mode."
        echo "Usage: bash run_phase2.sh patch <model_name> <split> <absolute_path_to_excel>"
        exit 1
    fi

    echo ">>> Running PATCH mode on $MINI_DATASET with model $MODEL_NAME"

    # --- Step 1: Trích xuất và chuẩn bị dữ liệu ---
    echo ">>> Step 1: Extracting and preparing data..."
    python3 - <<EOF
import pandas as pd
import numpy as np
import sys

excel_path = "$EXCEL_PATH"
base_tsv_path = "$BASE_TSV"
mini_tsv_path = "$MINI_TSV"

try:
    df_excel = pd.read_excel(excel_path)
    df_tsv = pd.read_csv(base_tsv_path, sep='\t')

    # Set index cho cả 2 DataFrame
    df_excel.set_index('index', inplace=True)
    df_tsv.set_index('index', inplace=True)

    # Lọc các dòng Lesion_Reasoning trong Excel
    mask = df_excel['category'] == 'Lesion_Reasoning'

    if mask.sum() == 0:
        print("No rows found with category 'Lesion_Reasoning' in Excel.")
        sys.exit(0)

    # Cập nhật cột question, answer, type từ TSV sang Excel (theo index)
    df_excel.loc[mask, 'question'] = df_tsv.loc[mask.index[mask], 'question']
    df_excel.loc[mask, 'answer'] = df_tsv.loc[mask.index[mask], 'answer']
    df_excel.loc[mask, 'type'] = df_tsv.loc[mask.index[mask], 'type']

    # Xóa prediction và score
    df_excel.loc[mask, 'prediction'] = np.nan
    if 'score' in df_excel.columns:
        df_excel.loc[mask, 'score'] = np.nan

    # Reset index và lưu lại Excel
    df_excel.reset_index(inplace=True)
    df_excel.to_excel(excel_path, index=False)

    # Tách các dòng Lesion_Reasoning từ TSV -> mini TSV
    df_mini = df_tsv.loc[mask.index[mask]].copy()
    df_mini.reset_index(inplace=True)
    df_mini.to_csv(mini_tsv_path, sep='\t', index=False)

    print(f"Data prepared successfully. Mini dataset saved to {mini_tsv_path}")
    print(f"Number of Lesion_Reasoning rows: {mask.sum()}")

except Exception as e:
    print(f"Error in Step 1: {e}")
    sys.exit(1)
EOF

    if [ $? -ne 0 ]; then
        echo "Step 1 failed."
        exit 1
    fi

    # --- Step 2: Chạy Inference trên Mini Dataset ---
    echo ">>> Step 2: Running Inference on Mini Dataset..."
    python run.py --data $MINI_DATASET --model "$MODEL_NAME" --mode infer --verbose --reuse --use-vllm

    if [ $? -ne 0 ]; then
        echo "Step 2 Inference failed."
        exit 1
    fi

    # --- Step 3: Gộp kết quả về file Excel gốc ---
    echo ">>> Step 3: Merging results back to original Excel file..."
    python3 - <<EOF
import pandas as pd
import glob
import os
import sys

excel_path = "$EXCEL_PATH"
model_name = "$MODEL_NAME"
mini_dataset = "$MINI_DATASET"

# Tìm file Excel mới nhất của mini dataset
search_pattern1 = f'outputs/{model_name}/*/*_{mini_dataset}.xlsx'
search_pattern2 = f'outputs/{model_name}/*_{mini_dataset}.xlsx'

files = glob.glob(search_pattern1) + glob.glob(search_pattern2)

if not files:
    print("Error: No output file found for mini dataset.")
    print(f"  Searched: {search_pattern1}")
    print(f"  Searched: {search_pattern2}")
    sys.exit(1)

latest_file = max(files, key=os.path.getmtime)
print(f"Found latest result file: {latest_file}")

try:
    df_goc = pd.read_excel(excel_path)
    df_moi = pd.read_excel(latest_file)

    # Set index và update
    df_goc.set_index('index', inplace=True)
    df_moi.set_index('index', inplace=True)

    df_goc.update(df_moi)

    df_goc.reset_index(inplace=True)
    df_goc.to_excel(excel_path, index=False)
    print(f"Successfully merged patch results into {excel_path}")

except Exception as e:
    print(f"Error in Step 3: {e}")
    sys.exit(1)
EOF

else
    echo "Error: Invalid mode '$MODE'. Must be 'full' or 'patch'."
    exit 1
fi