# Hướng Dẫn Chạy Phase 2 (DermNet VLM)

Cấp quyền thực thi trước khi chạy:
```bash
cd DermNet_Dataset/Phase_2/VLMEvalKit
chmod +x run_phase2.sh
```

---

## 1. CHẠY MỚI HOÀN TOÀN (FULL RUN)
Dành cho các tập dữ liệu chưa từng được chạy qua model.

**Nhóm 1.1: Chạy cả Val và Test (Các model chạy full)**
```bash
bash run_phase2.sh full Qwen3.5-35B-A3B val
bash run_phase2.sh full Qwen3.5-35B-A3B test

bash run_phase2.sh full Qwen3-VL-8B-Instruct val
bash run_phase2.sh full Qwen3-VL-8B-Instruct test

bash run_phase2.sh full LLaVA-med-v1.5-7B val
bash run_phase2.sh full LLaVA-med-v1.5-7B test
```

**Nhóm 1.2: Chỉ chạy Test (Vì tập Val đã chạy xong)**
```bash
bash run_phase2.sh full Vintern-1B-v2 test
bash run_phase2.sh full Vintern-3B-beta test
```

---

## 2. CHẠY VÁ LỖI LESION REASONING (PATCH RUN)
Dành cho các tập dữ liệu đã có file kết quả Excel, chỉ cần chạy lại riêng phần câu hỏi Lesion Reasoning mới. Script tự động nối kết quả đè vào file gốc.

**Nhóm 2.1: Vá tập Val (Dành cho các model đã chạy xong Val)**
```bash
bash run_phase2.sh patch deepseek_vl2 val outputs/deepseek_vl2/deepseek_vl2_int8_DermNet_Val_4k.xlsx
bash run_phase2.sh patch deepseek_vl2_small val outputs/deepseek_vl2_small/deepseek_vl2_small_DermNet_Val_4k.xlsx
bash run_phase2.sh patch deepseek_vl2_tiny val outputs/deepseek_vl2_tiny/deepseek_vl2_tiny_DermNet_Val_4k.xlsx
bash run_phase2.sh patch Vintern-1B-v2 val outputs/Vintern-1B-v2/Vintern-1B-v2_DermNet_Val_4k_mac.xlsx
bash run_phase2.sh patch Vintern-3B-beta val outputs/Vintern-3B-beta/Vintern-3B-beta_DermNet_Val_4k.xlsx
```

**Nhóm 2.2: Vá tập Test (Chỉ áp dụng cho họ Deepseek đã chạy xong Test)**
```bash
bash run_phase2.sh patch deepseek_vl2 test outputs/deepseek_vl2/deepseek_vl2_int8_DermNet_Test_1of3.xlsx
bash run_phase2.sh patch deepseek_vl2_small test outputs/deepseek_vl2_small/deepseek_vl2_small_DermNet_Test_1of3.xlsx
bash run_phase2.sh patch deepseek_vl2_tiny test outputs/deepseek_vl2_tiny/deepseek_vl2_tiny_DermNet_Test_1of3.xlsx
```
