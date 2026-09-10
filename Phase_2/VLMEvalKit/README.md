# DermNet Phase 2 Runner

Hướng dẫn đầy đủ nằm tại [README.md](../../README.md) ở root repository.

## Thứ tự chạy trên server

```text
1. Cài 4 requirement vào 4 environment riêng
2. Chuẩn bị source DeepSeek/Huatuo và file mapping
3. Submit lần lượt 4 run-group, mỗi job được cấp 2 GPU
4. Submit lại đúng run-group nếu bị gián đoạn
```

Không chạy cài environment bên trong compute job. Hoàn thành bước setup trước trên login/setup node theo [hướng dẫn cài server](../../docs/SERVER_SETUP.md).

## Bốn job inference

Chạy từ root repository:

```bash
# Job 1: Qwen + DeepSeek Small/Tiny
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm

# Job 2: DeepSeek-VL2 8-bit
bash Phase_2/VLMEvalKit/run_phase2.sh run-group deepseek-int8

# Job 3: Vintern 1B/3B
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern

# Job 4: HuatuoGPT-Vision-34B
bash Phase_2/VLMEvalKit/run_phase2.sh run-group huatuo
```

Mỗi nhóm dùng đúng environment được khai báo trong `.phase2-server-env.sh`. Chi tiết submit job: [SCHEDULER_RUN.md](../../docs/SCHEDULER_RUN.md).

## Chạy tiếp sau gián đoạn

Chạy lại đúng lệnh `run-group` đã bị dừng. Ví dụ:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

Runner bỏ qua lượt đã hoàn chỉnh và tiếp tục lượt còn thiếu hoặc thất bại.

## Kiểm tra kế hoạch

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
DRY_RUN=1 GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 \
  bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

## Vá riêng Lesion Reasoning

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh patch \
  deepseek_vl2_tiny \
  DermNet_Val_VI \
  /absolute/path/to/existing_result.xlsx
```

Lệnh `patch` giữ nguyên Excel nguồn, tạo bộ dữ liệu nhỏ chứa các dòng `Lesion_Reasoning`, chạy lại vào file riêng rồi mới gộp prediction mới. Nếu bị gián đoạn, chạy lại cùng lệnh để dùng checkpoint đã có.
