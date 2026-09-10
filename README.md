# DermNet Dataset

Runner Phase 2 chạy **8 model × 2 bộ dữ liệu tiếng Việt = 16 lượt** trên server 2 GPU × 96 GB. Điểm vào duy nhất:

```text
Phase_2/VLMEvalKit/run_phase2.sh
```

## Chạy riêng bốn nhóm — luồng chính

Sau khi bốn environment đã được chuẩn bị và có file `Phase_2/VLMEvalKit/.phase2-server-env.sh`, submit bốn job riêng. Mỗi job yêu cầu 2 GPU và chỉ gọi một lệnh inference:

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

Với Slurm, phần đầu mỗi job thường có:

```bash
#SBATCH --gres=gpu:2
cd /shared/path/DermNet_Dataset
```

Tên resource chính xác theo cấu hình của cụm máy. Trên server chỉ có hai GPU, có thể submit cả bốn job và để scheduler xếp chạy lần lượt.

### Cách runner dùng hai GPU

| Nhóm | Lượt | Cách dùng GPU |
|---|---:|---|
| `vllm` | 8 | Qwen dùng cả 2 GPU; DeepSeek chia 2 worker |
| `deepseek-int8` | 2 | Val và Test chạy song song, mỗi GPU một lượt |
| `vintern` | 4 | Hai worker, mỗi GPU xử lý một hàng đợi |
| `huatuo` | 2 | Val và Test chạy song song, mỗi GPU một lượt |

Runner giữ đúng `CUDA_VISIBLE_DEVICES` do scheduler cấp. Các lượt song song có work directory riêng nên không ghi đè trạng thái.

### Chạy tiếp sau gián đoạn

Submit lại đúng lệnh của nhóm bị dừng. Ví dụ:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

Runner kiểm tra output, bỏ qua lượt đã hoàn chỉnh và tiếp tục lượt thiếu hoặc thất bại. Khi cần chẩn đoán tuần tự:

```bash
RUN_GROUP_WORKERS=1 bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern
```

Hướng dẫn scheduler đầy đủ: [docs/SCHEDULER_RUN.md](docs/SCHEDULER_RUN.md).

## Cài riêng bốn environment

Cài environment một lần trên setup/login node trước khi submit compute job. Bốn profile được tách riêng vì phiên bản Torch, Transformers và backend khác nhau:

| Environment | Requirement | Model |
|---|---|---|
| `dermnet-vllm` | [vllm-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/vllm-blackwell.txt) | Qwen, DeepSeek Small/Tiny |
| `dermnet-deepseek-int8` | [deepseek-int8-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/deepseek-int8-blackwell.txt) | DeepSeek-VL2 8-bit |
| `dermnet-vintern` | [vintern-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/vintern-blackwell.txt) | Vintern 1B/3B |
| `dermnet-huatuo` | [huatuo-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/huatuo-blackwell.txt) | HuatuoGPT-Vision-34B |

Trên setup node có GPU, lệnh sau tạo đủ bốn env, cài source đã ghim, áp dụng bản vá Blackwell, tạo file mapping và chạy doctor:

```bash
git clone https://github.com/MinhKemm/DermNet_Dataset.git
cd DermNet_Dataset
bash Phase_2/VLMEvalKit/run_phase2.sh setup
```

Nếu quản trị viên cài từng env thủ công, dùng toàn bộ lệnh tại [docs/SERVER_SETUP.md](docs/SERVER_SETUP.md#3-setup-thủ-công-từng-environment). Quy trình này bao gồm:

1. Tạo bốn Conda environment và cài đúng bốn requirement ở bảng trên.
2. Clone đúng commit DeepSeek-VL2 và HuatuoGPT-Vision.
3. Áp dụng bản vá attention cho Blackwell.
4. Tạo `Phase_2/VLMEvalKit/.phase2-server-env.sh` để runner chọn đúng Python.

Nếu mapping nằm ngoài repository:

```bash
export SERVER_ENV_FILE=/shared/path/.phase2-server-env.sh
```

## Kế hoạch model và dữ liệu

Chỉ chạy `DermNet_Val_VI.tsv` và `DermNet_Test_VI.tsv`:

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

Hai bản DeepSeek vá sử dụng Excel nguồn được giữ nguyên. Runner tách các dòng cần sửa, inference vào file riêng rồi merge kết quả mới. Chi tiết: [docs/DEEPSEEK_RUN_PLAN.md](docs/DEEPSEEK_RUN_PLAN.md).

## Output và kiểm tra

- Kết quả: `Phase_2/VLMEvalKit/outputs/answer-format-v4-vllm/`.
- Log, checkpoint và trạng thái: `outputs/answer-format-v4-vllm/.phase2-runner/`.
- Work directory cho lượt full song song: `outputs/answer-format-v4-vllm/two-gpu-jobs/`.

Kiểm tra kế hoạch trước khi submit:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
DRY_RUN=1 GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 \
  bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

Tài liệu liên quan:

- [Cài environment server](docs/SERVER_SETUP.md)
- [Chạy bốn job scheduler](docs/SCHEDULER_RUN.md)
- [Backend vLLM/Blackwell](docs/VLLM_SERVER.md)
- [Kiểm tra dữ liệu](docs/DATASET_AUDIT.md)
- [Rà soát Excel vá](docs/RESULTS_PATCH_AUDIT.md)

Phương án phụ cho máy cho phép setup và inference trong cùng phiên:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh server
```
