# DermNet Phase 2 Runner

Hướng dẫn đầy đủ nằm tại [`README.md`](../../README.md) ở root repository.

## Chạy toàn bộ

Lần đầu trên server, từ root repository:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh server
```

Lệnh này setup môi trường, doctor rồi chạy toàn bộ. Nếu môi trường đã được setup, trong thư mục này có thể chạy thẳng:

```bash
bash run_phase2.sh all
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
bash run_phase2.sh full Vintern-1B-v2 DermNet_Val_VI
```

## Vá riêng Lesion Reasoning

```bash
bash run_phase2.sh patch \
  deepseek_vl2_tiny \
  DermNet_Val_VI \
  /absolute/path/to/existing_result.xlsx
```

Lệnh patch tạo mini dataset, chạy lại đúng các dòng `Lesion_Reasoning`, kiểm tra kết quả đầy đủ, sao lưu file cũ rồi mới gộp prediction mới. Nếu bị gián đoạn, chạy lại cùng lệnh để dùng checkpoint đã có.

Xem README tại root để biết bốn profile Python, hai dataset tiếng Việt, backend model, log và checkpoint.
