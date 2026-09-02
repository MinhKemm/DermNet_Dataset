# Hướng Dẫn Chạy Phase 2 (DermNet VLM)

Cấp quyền thực thi trước khi chạy:
```bash
cd /content/DermNet_Dataset/Phase_2/VLMEvalKit
chmod +x run_phase2.sh
```

---

## 1. CHẠY MỚI HOÀN TOÀN (FULL RUN)
Dành cho các tập dữ liệu chưa từng được chạy qua model.

**Nhóm 1.1: Chạy cả Val và Test (Các model chưa chạy tí nào)**
```bash
bash run_phase2.sh full Qwen3.6-35B val
bash run_phase2.sh full Qwen3.6-35B test

bash run_phase2.sh full Qwen3-VL-8B val
bash run_phase2.sh full Qwen3-VL-8B test

bash run_phase2.sh full LLaVA-med-v1.5-7B val
bash run_phase2.sh full LLaVA-med-v1.5-7B test
```

**Nhóm 1.2: Chỉ chạy Test (Vì tập Val đã chạy xong)**
```bash
bash run_phase2.sh full Vintern-1B test
bash run_phase2.sh full Vintern-3B test
```

---

## 2. CHẠY VÁ LỖI LESION REASONING (PATCH RUN)
Dành cho các tập dữ liệu đã có file kết quả Excel, chỉ cần chạy lại riêng phần câu hỏi Lesion Reasoning mới. Script tự động nối kết quả đè vào file gốc.

**Nhóm 2.1: Vá tập Val (Dành cho các model đã chạy xong Val)**
```bash
bash run_phase2.sh patch Deepseek-VL2-8bit val /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-8bit/Deepseek-VL2-8bit_DermNet_Val_4k-2_mac_relative.xlsx
bash run_phase2.sh patch Deepseek-VL2-small val /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-small/Deepseek-VL2-small_DermNet_Val_4k-2_mac_relative.xlsx
bash run_phase2.sh patch Deepseek-VL2-tiny-16bit val /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-tiny-16bit/Deepseek-VL2-tiny-16bit_DermNet_Val_4k-2_mac_relative.xlsx
bash run_phase2.sh patch Vintern-1B val /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Vintern-1B/Vintern-1B_DermNet_Val_4k-2_mac_relative.xlsx
bash run_phase2.sh patch Vintern-3B val /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Vintern-3B/Vintern-3B_DermNet_Val_4k-2_mac_relative.xlsx
```

**Nhóm 2.2: Vá tập Test (Chỉ áp dụng cho họ Deepseek đã chạy xong Test)**
```bash
bash run_phase2.sh patch Deepseek-VL2-8bit test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-8bit/Deepseek-VL2-8bit_DermNet_Test_mac_relative.xlsx
bash run_phase2.sh patch Deepseek-VL2-small test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-small/Deepseek-VL2-small_DermNet_Test_mac_relative.xlsx
bash run_phase2.sh patch Deepseek-VL2-tiny-16bit test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-tiny-16bit/Deepseek-VL2-tiny-16bit_DermNet_Test_mac_relative.xlsx
```
