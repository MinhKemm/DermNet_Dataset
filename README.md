# DermNet Dataset

## Chạy trên server

Chỉ cần chạy một file: `bash Phase_2/VLMEvalKit/run_phase2.sh all`. **LLaVA-med đã bỏ khỏi danh sách mặc định**, không cần biến SKIP_LLAVA. Xem [backend Blackwell/vLLM](docs/VLLM_SERVER.md). File shell dùng mã nguồn và dữ liệu trong checkout, không phải file độc lập có thể tách khỏi repo.

Danh sách hiện tại có **8 model**, chỉ dùng Val và Test tiếng Việt: **16 lượt = 12 full + 4 vá**. Một lượt là một model chạy trên một bộ dữ liệu. Các lượt chạy tuần tự.

| Model | Val VI | Test VI |
|---|---|---|
| DeepSeek-VL2-small | Full | Full |
| DeepSeek-VL2 8-bit | Vá | Vá |
| DeepSeek-VL2-tiny 16-bit (BF16) | Vá | Vá |
| Qwen3.5-35B-A3B | Full | Full |
| Qwen3-VL-8B-Instruct | Full | Full |
| HuatuoGPT-Vision-34B | Full | Full |
| Vintern-1B-v2 | Full mới | Full mới |
| Vintern-3B-beta | Full mới | Full mới |

Sau khi kích hoạt environment Python:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
bash Phase_2/VLMEvalKit/run_phase2.sh all
# Chạy tiếp sau gián đoạn:
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Manifest chính xác: [dermnet_jobs.txt](Phase_2/VLMEvalKit/scripts/dermnet_jobs.txt). Server dự kiến 2 GPU × 96 GB. Runner chọn model theo ngưỡng cấu hình; không mặc định cộng VRAM hai GPU thành bộ nhớ một model.

## Vintern chạy lại mới

Cả hai Vintern chạy full Val và Test, không vá từ Excel cũ. Kết quả riêng tại `outputs/answer-format-v4-vllm/vintern-full-rerun-20260908`, không dùng checkpoint trước đợt này. Lệnh resume giữ phần đã hoàn thành của đợt mới, không xóa lại mỗi lần chạy.

Kiểm tra checkout hiện tại không tìm thấy Excel/checkpoint Vintern cũ để xóa. Không xóa mã model hay kết quả model khác. Chưa kiểm tra hoặc xóa file trên server từ xa.

## DeepSeek chạy vá

Bốn Excel người dùng cung cấp được giữ nguyên nội dung trong `Phase_2/VLMEvalKit/outputs/deepseek_vl2_int8/source/` và `outputs/deepseek_vl2_tiny/source/`. Val có 4.000 dòng; Test có 19.133 dòng. Đã kiểm tra tương thích index, ảnh và metadata.

Small dùng `deepseek-ai/deepseek-vl2-small`; Tiny dùng `deepseek-ai/deepseek-vl2-tiny`, BF16 không lượng tử hóa; 8-bit dùng `deepseek-ai/deepseek-vl2` với `load_in_8bit=True`. Một số lớp của bản 8-bit được adapter giữ/khôi phục BF16.

Luồng vá: kiểm tra nguồn → tách reasoning và các sửa dữ liệu được cho phép → chạy model → merge Excel riêng, giữ nguyên nguồn và xóa score cũ của dòng cập nhật. Nhật ký sửa câu hỏi: `scripts/dataset_repairs.json`. Chi tiết nguồn: [DeepSeek](docs/DEEPSEEK_RUN_PLAN.md).

Gemma, Huatuo 7B, InternVL, Janus, LLaVA 1.5 7B 4-bit, Phi, Qwen2.5-VL và SmolVLM đã bỏ khỏi danh sách tự động. File lịch sử của các model này và dữ liệu Anh không bị xóa.

## Environment

Quy trình cài từ đầu: **[Setup server](docs/SERVER_SETUP.md)** — tạo environment theo nhóm model, cài PyTorch/CUDA, package, cấu hình Python và kiểm tra trước khi chạy. Đây là hướng dẫn chuẩn bị, chưa phải lockfile đã kiểm chứng trên GPU server.

Vintern hỗ trợ `PYTHON_VINTERN` riêng; nếu không đặt, runner dùng `PYTHON_LEGACY`. Các lệnh cài đặt chạy một lần bởi quản trị viên, sau đó dùng cùng cấu hình cho `all` và `resume`.

Người quản trị chuẩn bị Python/PyTorch/CUDA, pandas, openpyxl và package tương ứng với mỗi model. Có thể đặt `PYTHON_BIN` chung hoặc `PYTHON_QWEN`, `PYTHON_LEGACY`, `PYTHON_DEEPSEEK`, `PYTHON_HUATUO`.

Huatuo dùng [mã inference chính thức](https://github.com/FreedomIntelligence/HuatuoGPT-Vision):

```bash
export HUATUO_SOURCE_DIR=/srv/HuatuoGPT-Vision
export PYTHON_HUATUO=/env/huatuo/bin/python
```

Huatuo vẫn cần module llava trong repo tác giả; bỏ model LLaVA-med không có nghĩa xóa dependency nội bộ của Huatuo. Khi cần xác thực tải model, đặt `HF_TOKEN` trong environment.

## Dữ liệu, checkpoint và kiểm tra

Hai dataset tiếng Việt được sử dụng trong LMUData:

- `DermNet_Val_VI.tsv`
- `DermNet_Test_VI.tsv`

Prompt dùng cột type: Multi_choice trả chữ cái; Judgement trả Có/Không hoặc Yes/No; Fill_in_blank trả cụm từ thiếu; Short_answer trả cụm từ hoặc câu ngắn.

Mặc định `MISSING_IMAGE_POLICY=fail`: thiếu ảnh sẽ dừng. `skip` chỉ dùng khi chấp nhận bỏ dòng, có log danh sách ảnh thiếu. [Báo cáo dataset](docs/DATASET_AUDIT.md) ghi những lỗi còn tồn tại.

Kết quả mới: `outputs/answer-format-v4-vllm`. Trạng thái/log/backup/mini dataset: `outputs/answer-format-v4-vllm/.phase2-runner`. Có thể đặt `RUN_WORK_DIR` riêng và giữ nguyên khi resume. Runner đang dùng `--mode infer`, chưa tự chấm điểm; score cũ của reasoning được xóa khi merge.

Kiểm thử cục bộ:

```bash
cd Phase_2/VLMEvalKit
python -m unittest tests.test_legacy_model_loading tests.test_dermnet_runner tests.test_dermnet_reasoning_patch tests.test_dermnet_prompt tests.test_dermnet_dataset_contract tests.test_deepseek_vl2_instruction tests.test_huatuo_vision
bash -n run_phase2.sh
```

Đã kiểm tra tương thích 4 nguồn patch; chưa chạy suy luận model trên server GPU. Chi tiết: [rà soát Excel và kế hoạch](docs/RESULTS_PATCH_AUDIT.md).
