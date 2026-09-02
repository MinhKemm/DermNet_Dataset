# Hướng dẫn chạy đánh giá VLM (Phase 2)

Hướng dẫn chạy đánh giá (benchmark) các Vision-Language Models cho dự án DermNet thông qua script `run_phase2.sh`.

## Setup ban đầu
Trước khi chạy lần đầu, hãy cấp quyền thực thi cho script bash:
```bash
chmod +x run_phase2.sh
```

---

## Mục 1: Chạy mới hoàn toàn (Full Run)
Dùng để chạy đánh giá trên toàn bộ tập dữ liệu từ đầu.
Cú pháp cơ bản: `./run_phase2.sh full <model_name> <split>`

**Ví dụ chạy cho tập Test:**
```bash
./run_phase2.sh full Qwen3-VL-8B test
./run_phase2.sh full Qwen3.6-35B test
```

**Ví dụ chạy cho tập Val:**
```bash
./run_phase2.sh full Qwen3-VL-8B val
./run_phase2.sh full Qwen3.6-35B val
```

---

## Mục 2: Chạy vá lỗi (Patch Run)
Dùng trong trường hợp bạn cần chạy lại và cập nhật nhóm câu hỏi `Lesion_Reasoning` (do fix prompt, bổ sung dữ liệu...) mà không phải đánh giá lại toàn bộ file. Quá trình này sẽ diễn ra tự động 3 bước:
1. Cập nhật câu hỏi và lấy ra bộ test mini (`..._Reasoning_Fix.tsv`).
2. Chạy evaluation trên bộ mini này (cờ `--reuse`).
3. Gộp ngược kết quả mới vào file Excel gốc.

Cú pháp cơ bản: `./run_phase2.sh patch <model_name> <split> <absolute_path_to_excel>`

- `<split>`: Tham số này phải là `test` hoặc `val`.
- `<absolute_path_to_excel>`: Đường dẫn tuyệt đối trỏ tới file `.xlsx` mà hệ thống đã xuất ra trước đó của mô hình.

**Ví dụ các lệnh chạy Patch sẵn có trên tập `test`:**
*(Lưu ý: Nếu đường dẫn file lưu ở nơi khác, hãy thay đoạn `/content/DermNet_Dataset/...` bằng đường dẫn thực tế)*

```bash
./run_phase2.sh patch Vintern-1B test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Vintern-1B/Vintern-1B_DermNet_Test_mac_relative.xlsx

./run_phase2.sh patch Vintern-3B test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Vintern-3B/Vintern-3B_DermNet_Test_mac_relative.xlsx

./run_phase2.sh patch LLaVA-med-v1.5-7B test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/LLaVA-med-v1.5-7B/LLaVA-med-v1.5-7B_DermNet_Test_mac_relative.xlsx

./run_phase2.sh patch Deepseek-VL2-8bit test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-8bit/Deepseek-VL2-8bit_DermNet_Test_mac_relative.xlsx

./run_phase2.sh patch Deepseek-VL2-small test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-small/Deepseek-VL2-small_DermNet_Test_mac_relative.xlsx

./run_phase2.sh patch Deepseek-VL2-tiny-16bit test /content/DermNet_Dataset/Phase_2/VLMEvalKit/outputs/Deepseek-VL2-tiny-16bit/Deepseek-VL2-tiny-16bit_DermNet_Test_mac_relative.xlsx
```
