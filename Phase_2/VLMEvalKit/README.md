# DermNet Phase 2 Runner

Hướng dẫn đầy đủ nằm tại [`README.md`](../../README.md) ở root repository.

## Chạy toàn bộ

Trong thư mục này:

```bash
bash run_phase2.sh all
```

Hoặc từ root repository:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

## Chạy tiếp sau khi gián đoạn

```bash
bash run_phase2.sh resume
```

## Kiểm tra kế hoạch

```bash
DRY_RUN=1 bash run_phase2.sh plan
```

## Chạy riêng một job

```bash
bash run_phase2.sh full Vintern-1B-v2 DermNet_Val_4k-2_mac_relative
```

Runner sử dụng environment Python đã được chuẩn bị trên server. Xem README tại root để biết các biến `PYTHON_BIN`, `PYTHON_QWEN`, `PYTHON_LEGACY`, `PYTHON_DEEPSEEK`, profile GPU, dữ liệu song ngữ, log và checkpoint.
