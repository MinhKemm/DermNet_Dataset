# DermNet Dataset

## Chạy một file trên server

Sau khi kích hoạt environment Python đã chuẩn bị, tại root repository:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Nếu bị gián đoạn, dùng lại cùng cấu hình và thư mục kết quả:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

`all/resume` chạy đúng kế hoạch tiếng Việt cũ, tối đa **8 lượt full + 8 lượt patch**. Checkpoint giúp tiếp tục phần chưa xong. Một file shell điều phối các module Python có sẵn trong repository.

## Danh sách công việc

| Model | Val | Test |
|---|---|---|
| Qwen3.5-35B-A3B | Full | Full |
| Qwen3-VL-8B-Instruct | Full | Full |
| LLaVA-med-v1.5-7B | Full | Full |
| Vintern-1B-v2 | Patch reasoning | Full |
| Vintern-3B-beta | Patch reasoning | Full |
| deepseek_vl2 | Patch reasoning | Patch reasoning |
| deepseek_vl2_small | Patch reasoning | Patch reasoning |
| deepseek_vl2_tiny | Patch reasoning | Patch reasoning |

Val dùng `DermNet_Val_4k-2_mac_relative`; Test dùng `DermNet_Test_mac_relative`.

Mặc định `MODEL_PROFILE=auto` lọc model theo VRAM ước tính, ghi rõ model bị bỏ qua. Không tự đổi sang biến thể INT4/INT8 khác. `MODEL_PROFILE=full` chọn đủ 16 lượt, chỉ dùng khi server đủ tài nguyên. Ngưỡng VRAM không bảo đảm model sẽ chạy vừa trong mọi environment.

## Chuẩn bị file kết quả cũ để vá

Clone code chưa đủ để vá: cần chép 8 file kết quả cũ lên server. Mặc định chúng nằm dưới `Phase_2/VLMEvalKit/outputs`, hoặc đặt `LEGACY_RESULTS_DIR` trỏ tới thư mục chứa các thư mục model:

```text
outputs/
├── deepseek_vl2/
│   ├── deepseek_vl2_int8_DermNet_Val_4k.xlsx
│   └── deepseek_vl2_int8_DermNet_Test_1of3.xlsx
├── deepseek_vl2_small/
│   ├── deepseek_vl2_small_DermNet_Val_4k.xlsx
│   └── deepseek_vl2_small_DermNet_Test_1of3.xlsx
├── deepseek_vl2_tiny/
│   ├── deepseek_vl2_tiny_DermNet_Val_4k.xlsx
│   └── deepseek_vl2_tiny_DermNet_Test_1of3.xlsx
├── Vintern-1B-v2/
│   └── Vintern-1B-v2_DermNet_Val_4k_mac.xlsx
└── Vintern-3B-beta/
    └── Vintern-3B-beta_DermNet_Val_4k.xlsx
```

Tên file DeepSeek lớn giữ theo README cũ, kể cả hậu tố `int8`; model được gọi vẫn là `deepseek_vl2` như lệnh cũ. Cần bảo đảm file thực tế đúng model/thí nghiệm mong muốn trước khi gộp.

```bash
LEGACY_RESULTS_DIR=/srv/dermnet/old-results bash Phase_2/VLMEvalKit/run_phase2.sh all
LEGACY_RESULTS_DIR=/srv/dermnet/old-results bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Runner kiểm tra sự tồn tại của toàn bộ file patch thuộc model được chọn trước khi bắt đầu inference. Thiếu file sẽ liệt kê và dừng. Không tự chuyển patch thành lượt full.

## Environment và dữ liệu

Server cần Linux/Bash, NVIDIA/CUDA hoạt động, trọng số model hoặc quyền tải từ Hugging Face, cùng Python đã cài package của các model. Qwen cần vLLM; LLaVA/DeepSeek cần package tương ứng; đọc/ghi Excel cần openpyxl.

Có thể chỉ định Python cho từng nhóm:

```bash
PYTHON_QWEN=/env/qwen/bin/python \
PYTHON_LEGACY=/env/legacy/bin/python \
PYTHON_DEEPSEEK=/env/deepseek/bin/python \
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Mặc định dùng `python3`; đặt `PYTHON_BIN` để đổi chung. Nếu model yêu cầu xác thực, đặt `HF_TOKEN` trong environment.

Mặc định thiếu ảnh sẽ dừng trước inference. Chỉ dùng `MISSING_IMAGE_POLICY=skip` nếu chấp nhận bộ kết quả thiếu dòng; các dòng bị bỏ được ghi trong `*.missing-images.txt`. Dataset runtime chuẩn hóa đường dẫn ảnh, không thay thế dataset nguồn.

## Kết quả và chạy tiếp

- Lượt full với prompt mới: `Phase_2/VLMEvalKit/outputs/answer-format-v2`.
- Log, marker, mini dataset, checkpoint patch và backup: `outputs/answer-format-v2/.phase2-runner`.
- Patch sao lưu rồi cập nhật trực tiếp file kết quả cũ đã chỉ định.
- `MAX_JOB_RETRIES=2` mặc định thử inference tối đa hai lần mỗi lượt.
- `RUN_WORK_DIR=/absolute/path` đặt thư mục thí nghiệm khác; giữ nguyên khi resume.
- `status` hiển thị marker full; trạng thái patch được kiểm tra lại khi resume.

Patch chỉ thay các dòng Lesion_Reasoning sau khi có đủ prediction hợp lệ; câu trả lời trống, lỗi và thiếu ảnh bị từ chối. Score cũ của các dòng được vá bị xóa và cần chấm lại. Runner đang chạy `--mode infer`, chưa tự tính điểm benchmark.

## Các lệnh riêng

```bash
# Xem lệnh mà không chạy model; all vẫn yêu cầu file patch tồn tại
DRY_RUN=1 bash Phase_2/VLMEvalKit/run_phase2.sh all

# Chạy tiếng Anh khi cần, ngoài kế hoạch mặc định
bash Phase_2/VLMEvalKit/run_phase2.sh full Vintern-1B-v2 DermNet_Val_4k_en

# Vá riêng; chạy lại cùng lệnh để tiếp tục
bash Phase_2/VLMEvalKit/run_phase2.sh patch deepseek_vl2_tiny DermNet_Val_4k-2_mac_relative /absolute/path/result.xlsx
```

## Cách trả lời

| Type | Định dạng |
|---|---|
| Multi_choice | Chữ cái in hoa: A hoặc ACD |
| Judgement | Có/Không cho Việt; Yes/No cho Anh |
| Fill_in_blank | Thuật ngữ hoặc cụm từ thiếu |
| Short_answer | Cụm từ hoặc một câu ngắn |

Xem [báo cáo rà soát](docs/MODEL_ANSWER_REVIEW.md) về sửa prompt, kiểm thử và các câu hỏi nguồn còn cần duyệt. Chưa kiểm chứng suy luận thực tế trên GPU server.
