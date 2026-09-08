# DermNet Dataset

Repository dùng để chạy benchmark các Vision-Language Model trên bộ câu hỏi da liễu DermNet bằng tiếng Việt và tiếng Anh.

Toàn bộ quá trình inference được điều phối bằng đúng một file:

```text
Phase_2/VLMEvalKit/run_phase2.sh
```
Sau khi kích hoạt environment Python đã chuẩn bị trên server, dùng `run_phase2.sh` để chạy toàn bộ benchmark.

## Chạy nhanh

Tại thư mục root của repository:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Nếu quá trình bị dừng, mất kết nối SSH hoặc server khởi động lại, chạy:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Script sử dụng `--reuse --reuse-aux infer`, do đó dữ liệu inference đã có được tái sử dụng và các job chưa hoàn thành tiếp tục chạy.

## Yêu cầu phía server

Người quản trị server cần chuẩn bị trước:

- Linux và Bash.
- NVIDIA GPU hoạt động bình thường.
- Lệnh `nvidia-smi` sử dụng được.
- Environment Python đã cài PyTorch, Transformers, pandas và các package cần thiết cho model.
- Đủ dung lượng để tải model và lưu kết quả.
- Kết nối tới Hugging Face nếu model chưa được tải về máy.

Kiểm tra nhanh:

```bash
nvidia-smi
python3 -c "import torch, transformers, pandas; print(torch.__version__, transformers.__version__)"
```

Các model Qwen chạy qua vLLM nên environment tương ứng phải import được `vllm`. LLaVA và DeepSeek-VL2 phải import được package model của chúng. Runner kiểm tra các import cần thiết trước khi bắt đầu và dừng sớm nếu environment chưa đầy đủ.

## Các bộ dữ liệu được chạy

Mỗi model được chạy trên bốn dataset:

| Dataset | Ngôn ngữ | Phần dữ liệu |
|---|---|---|
| `DermNet_Val_4k-2_mac_relative` | Tiếng Việt | Validation |
| `DermNet_Test_mac_relative` | Tiếng Việt | Test |
| `DermNet_Val_4k_en` | Tiếng Anh | Validation |
| `DermNet_Test_1of3_en` | Tiếng Anh | Test |

Bốn file TSV tương ứng phải tồn tại trong `Phase_2/VLMEvalKit/LMUData` trước khi chạy.

## Danh sách model đầy đủ

Profile đầy đủ gồm tám model:

1. `Qwen3.5-35B-A3B`
2. `Qwen3-VL-8B-Instruct`
3. `LLaVA-med-v1.5-7B`
4. `Vintern-1B-v2`
5. `Vintern-3B-beta`
6. `deepseek_vl2_tiny`
7. `deepseek_vl2_small`
8. Một biến thể DeepSeek lớn được chọn theo VRAM

Khi server đủ cấu hình, tổng số lượt chạy là:

```text
8 model x 4 dataset = 32 job
```

## Tự chọn model theo GPU

Mặc định `MODEL_PROFILE=auto`. Script đọc VRAM bằng `nvidia-smi` và bỏ qua những model vượt quá cấu hình dự kiến.

| VRAM lớn nhất | DeepSeek được chọn | Ghi chú |
|---:|---|---|
| Từ 64 GB | `deepseek_vl2` | Bản đầy đủ |
| Từ 36 GB | `deepseek_vl2_int8` | Bản lượng tử 8-bit |
| Từ 22 GB | `deepseek_vl2_int4` | Bản lượng tử 4-bit |
| Dưới 22 GB | Không chạy biến thể DeepSeek lớn | Vẫn có thể chạy model nhỏ phù hợp |

Các ngưỡng trên là ngưỡng an toàn ước tính, không phải cam kết tuyệt đối. Mức sử dụng thực tế còn phụ thuộc CUDA, phiên bản thư viện, độ dài đầu ra và VRAM đang bị tiến trình khác chiếm dụng.

Các profile hỗ trợ:

```bash
# Tự chọn theo VRAM, khuyến nghị
MODEL_PROFILE=auto bash Phase_2/VLMEvalKit/run_phase2.sh all

# Ép chạy đủ tám model
MODEL_PROFILE=full bash Phase_2/VLMEvalKit/run_phase2.sh all

# Dùng DeepSeek INT8 và bỏ model lớn nhất
MODEL_PROFILE=balanced bash Phase_2/VLMEvalKit/run_phase2.sh all

# Chỉ chạy nhóm nhỏ và DeepSeek INT4
MODEL_PROFILE=minimal bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Không nên ép `MODEL_PROFILE=full` nếu server không đủ VRAM vì tiến trình có thể bị lỗi Out Of Memory.

## Chọn Python environment

Mặc định runner gọi `python3`:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Nếu environment sử dụng Python tại đường dẫn khác:

```bash
PYTHON_BIN=/path/to/env/bin/python bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Nếu tất cả model đã chạy được trong cùng một environment thì chỉ cần `PYTHON_BIN`.

Nếu người quản trị chuẩn bị environment riêng cho từng nhóm model:

```bash
PYTHON_QWEN=/env/qwen/bin/python \
PYTHON_LEGACY=/env/llava-vintern/bin/python \
PYTHON_DEEPSEEK=/env/deepseek/bin/python \
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Các đường dẫn trên giúp runner chọn đúng Python cho từng nhóm model.

## Kiểm tra trước khi chạy

In danh sách model, dataset và thứ tự job mà không chạy inference:

```bash
DRY_RUN=1 bash Phase_2/VLMEvalKit/run_phase2.sh plan
```

Xem trạng thái:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh status
```

Chạy riêng một model trên một dataset:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh full Vintern-1B-v2 DermNet_Val_4k-2_mac_relative
```

## Ảnh bị thiếu hoặc sai đường dẫn

Một số đường dẫn ảnh có thể khác nhau do chuẩn Unicode giữa Windows và Linux. Runner tạo bản TSV runtime đã chuẩn hóa tại:

```text
Phase_2/VLMEvalKit/outputs/.phase2-runner/datasets
```

Mặc định, dòng không tìm thấy ảnh được bỏ khỏi bản runtime và ghi vào file `*.missing-images.txt`. Dataset nguồn trong `Phase_2/VLMEvalKit/LMUData` không bị sửa.

Muốn dừng toàn bộ ngay khi phát hiện ảnh thiếu:

```bash
MISSING_IMAGE_POLICY=fail bash Phase_2/VLMEvalKit/run_phase2.sh all
```

## Log và kết quả

Log riêng của từng model/dataset được lưu tại:

```text
Phase_2/VLMEvalKit/outputs/.phase2-runner/logs
```

Kết quả inference được lưu dưới `Phase_2/VLMEvalKit/outputs`. File đánh dấu job hoàn thành nằm trong:

```text
Phase_2/VLMEvalKit/outputs/.phase2-runner/completed
```

Nếu một job lỗi, runner giữ checkpoint, tiếp tục thử theo `MAX_JOB_RETRIES`, rồi chuyển sang job tiếp theo. Mặc định mỗi job được thử tối đa hai lần:

```bash
MAX_JOB_RETRIES=3 bash Phase_2/VLMEvalKit/run_phase2.sh all
```

## Hugging Face token

Không ghi token trực tiếp vào file shell và không commit token lên Git.

Nếu model yêu cầu xác thực:

```bash
export HF_TOKEN="YOUR_TOKEN"
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

## Quy trình đề xuất trên server

```bash
# 1. Clone repository
git clone <repository-url> Dermnet-QA
cd Dermnet-QA

# 2. Kích hoạt environment do quản trị viên chuẩn bị
source /path/to/env/bin/activate

# 3. Kiểm tra kế hoạch
DRY_RUN=1 bash Phase_2/VLMEvalKit/run_phase2.sh plan

# 4. Chạy toàn bộ
bash Phase_2/VLMEvalKit/run_phase2.sh all

# 5. Nếu bị gián đoạn, chạy tiếp
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

## Cấu trúc chính

```text
DermNet_Dataset/
├── README.md
├── dermnet-output/                 # Ảnh DermNet
└── Phase_2/VLMEvalKit/
    ├── run_phase2.sh               # File duy nhất cần chạy
    ├── run.py                      # Entry point inference của VLMEvalKit
    ├── requirements.txt
    ├── LMUData/                    # TSV tiếng Việt và tiếng Anh
    └── outputs/                    # Log, checkpoint và kết quả
```
