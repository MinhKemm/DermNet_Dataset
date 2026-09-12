# DermNet Dataset

Phase 2 chạy **8 model × 2 bộ dữ liệu tiếng Việt = 16 lượt** trên server 2 GPU × 96 GB. Toàn bộ inference đi qua một file:

```text
Phase_2/VLMEvalKit/run_phase2.sh
```

## Quy trình chính trên server

Thực hiện theo đúng thứ tự:

```text
1. Clone repository
2. Cài 4 requirement vào 4 environment riêng
3. Chạy một lệnh chuẩn bị runtime
4. Submit lần lượt 4 run-group
5. Submit lại đúng run-group nếu bị gián đoạn
```

### Bước 1 — Clone repository

```bash
git clone https://github.com/MinhKemm/DermNet_Dataset.git
cd DermNet_Dataset
```

Repository, environment, dữ liệu và output cần nằm trên filesystem mà compute node đọc được.

### Bước 2 — Cài bốn requirement riêng

Mỗi job sử dụng một environment riêng:

| Job | Environment | Requirement |
|---|---|---|
| `vllm` | `dermnet-vllm` | [vllm-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/vllm-blackwell.txt) |
| `deepseek-int8` | `dermnet-deepseek-int8` | [deepseek-int8-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/deepseek-int8-blackwell.txt) |
| `vintern` | `dermnet-vintern` | [vintern-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/vintern-blackwell.txt) |
| `huatuo` | `dermnet-huatuo` | [huatuo-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/huatuo-blackwell.txt) |

Đứng tại root repository và chạy tuần tự:

```bash
export DERMNET_KIT_DIR="$PWD/Phase_2/VLMEvalKit"
export LEGACY_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

# Requirement cho job 1: vllm
conda create -n dermnet-vllm python=3.10 pip -y
conda run -n dermnet-vllm python -m pip install \
  -r "$DERMNET_KIT_DIR/requirements/server/vllm-blackwell.txt"
conda run -n dermnet-vllm python -m pip install --no-deps -e "$DERMNET_KIT_DIR"
conda run -n dermnet-vllm python -m pip check

# Requirement cho job 2: deepseek-int8
conda create -n dermnet-deepseek-int8 python=3.10 pip -y
conda run -n dermnet-deepseek-int8 python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url "$LEGACY_TORCH_INDEX_URL"
conda run -n dermnet-deepseek-int8 python -m pip install \
  -r "$DERMNET_KIT_DIR/requirements/server/deepseek-int8-blackwell.txt"
conda run -n dermnet-deepseek-int8 python -m pip install --no-deps -e "$DERMNET_KIT_DIR"
conda run -n dermnet-deepseek-int8 python -m pip check

# Requirement cho job 3: vintern
conda create -n dermnet-vintern python=3.10 pip -y
conda run -n dermnet-vintern python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url "$LEGACY_TORCH_INDEX_URL"
conda run -n dermnet-vintern python -m pip install \
  -r "$DERMNET_KIT_DIR/requirements/server/vintern-blackwell.txt"
conda run -n dermnet-vintern python -m pip install --no-deps -e "$DERMNET_KIT_DIR"
conda run -n dermnet-vintern python -m pip check

# Requirement cho job 4: huatuo
conda create -n dermnet-huatuo python=3.10 pip -y
conda run -n dermnet-huatuo python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url "$LEGACY_TORCH_INDEX_URL"
conda run -n dermnet-huatuo python -m pip install \
  -r "$DERMNET_KIT_DIR/requirements/server/huatuo-blackwell.txt"
conda run -n dermnet-huatuo python -m pip install --no-deps -e "$DERMNET_KIT_DIR"
conda run -n dermnet-huatuo python -m pip check
```

Các file server nhìn ngắn vì dòng đầu `-r ../../requirements.txt` nạp thêm [71 dependency lõi](Phase_2/VLMEvalKit/requirements.txt). Mỗi file sau đó khóa phiên bản riêng của backend; pip tiếp tục cài các package phụ cần thiết.

### Bước 3 — Chuẩn bị runtime một lần

Sau khi admin đã cài đủ bốn environment ở Bước 2, chạy:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh prepare-runtime
```

Lệnh này **không cài package, không chạy model và không cần GPU**. Nó xác nhận bốn environment theo tên chuẩn tồn tại, chạy `pip check`, tải đúng source DeepSeek/Huatuo, áp dụng bản vá Blackwell và tạo `Phase_2/VLMEvalKit/.phase2-server-env.sh`. Từ lần sau không cần làm lại Bước 3.

Tên environment và thư mục source khác mặc định có thể truyền bằng biến môi trường theo [hướng dẫn server](docs/SERVER_SETUP.md). Phần setup thủ công dài chỉ là phương án dự phòng.

Nếu file mapping nằm ngoài repository:

```bash
export SERVER_ENV_FILE=/shared/path/.phase2-server-env.sh
```

Nếu muốn script tự cài luôn bốn environment và node setup nhìn thấy GPU, có thể thay toàn bộ Bước 2 và 3 bằng:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh setup
```

### Bước 4 — Chạy riêng bốn job

Mỗi job yêu cầu 2 GPU. Trên server có đúng hai GPU, để scheduler chạy từng job theo thứ tự dưới đây:

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

Phần đầu Slurm job thường có:

```bash
#SBATCH --gres=gpu:2
cd /shared/path/DermNet_Dataset
```

| Job | Lượt | Cách dùng hai GPU |
|---|---:|---|
| `vllm` | 8 | Qwen dùng cả hai GPU; DeepSeek chia hai worker |
| `deepseek-int8` | 2 | Val và Test chạy song song |
| `vintern` | 4 | Hai worker, mỗi GPU một hàng đợi |
| `huatuo` | 2 | Val và Test chạy song song |

Runner giữ `CUDA_VISIBLE_DEVICES` do scheduler cấp và tách work directory cho các lượt chạy song song.

### Bước 5 — Chạy tiếp sau gián đoạn

Submit lại đúng lệnh của nhóm bị dừng. Runner bỏ qua lượt đã hoàn chỉnh và tiếp tục phần thiếu hoặc thất bại:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

Khi cần chạy tuần tự để chẩn đoán:

```bash
RUN_GROUP_WORKERS=1 bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern
```

## Model và dữ liệu

Chỉ sử dụng `DermNet_Val_VI.tsv` và `DermNet_Test_VI.tsv`:

| Model | Val VI | Test VI |
|---|---|---|
| Qwen3.5-35B-A3B | Full | Full |
| Qwen3-VL-8B-Instruct | Full | Full |
| HuatuoGPT-Vision-34B | Full | Full |
| Vintern-1B-v2 | Full mới | Full mới |
| Vintern-3B-beta | Full mới | Full mới |
| DeepSeek-VL2-small | Full | Full |
| DeepSeek-VL2 8-bit | Vá | Vá |
| DeepSeek-VL2-tiny BF16 | Vá | Vá |

Tổng cộng **12 lượt full + 4 lượt vá**. Manifest: [dermnet_jobs.txt](Phase_2/VLMEvalKit/scripts/dermnet_jobs.txt).

Hai bản DeepSeek vá giữ nguyên Excel nguồn, inference các dòng cần sửa vào file riêng rồi merge kết quả mới. Chi tiết: [docs/DEEPSEEK_RUN_PLAN.md](docs/DEEPSEEK_RUN_PLAN.md).

## Output và kiểm tra

- Kết quả: `Phase_2/VLMEvalKit/outputs/answer-format-v4-vllm/`.
- Log/checkpoint: `outputs/answer-format-v4-vllm/.phase2-runner/`.
- Lượt full song song: `outputs/answer-format-v4-vllm/two-gpu-jobs/`.

Kiểm tra kế hoạch:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
DRY_RUN=1 GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 \
  bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

Tài liệu chi tiết:

- [Cài environment server](docs/SERVER_SETUP.md)
- [Chạy scheduler](docs/SCHEDULER_RUN.md)
- [Backend vLLM/Blackwell](docs/VLLM_SERVER.md)
- [Kiểm tra dữ liệu](docs/DATASET_AUDIT.md)
- [Rà soát Excel vá](docs/RESULTS_PATCH_AUDIT.md)
