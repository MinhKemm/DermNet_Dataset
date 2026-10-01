# Chạy bốn nhóm model trên scheduler

Tài liệu này dành cho hệ thống HPC không cho `conda create` hoặc `pip install` trong compute job. Nguyên tắc là:

```text
Login/setup node: cài environment rồi chạy prepare-runtime một lần
Compute job:       chỉ kiểm tra environment được cấp rồi chạy inference
```

Không dùng lệnh `server` trong nội dung submit job vì `server` luôn gọi `setup`. Thay vào đó, chuẩn bị environment trước rồi dùng bốn lệnh `run-group` bên dưới.

## 1. Chuẩn bị một lần trước khi submit

Sau khi quản trị viên cài đủ bốn environment theo [hướng dẫn server](SERVER_SETUP.md), chạy:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh prepare-runtime
```

Lệnh này không cài package và không cần GPU. Sau khi hoàn tất phải có file:

```text
Phase_2/VLMEvalKit/.phase2-server-env.sh
```

Repository, bốn Conda environment, thư mục `vendor` và file mapping phải nằm trên filesystem mà compute node đọc được. Nếu scheduler chạy một bản clone khác, trỏ tới mapping dùng chung:

```bash
export SERVER_ENV_FILE=/shared/path/.phase2-server-env.sh
```

Login node không có GPU thì không cần chạy doctor tại đó. Mỗi `run-group` sẽ kiểm tra CUDA và dependency của riêng nhóm trên compute node trước khi bắt đầu model.

## 2. Bốn nhóm cần submit

Đứng tại root repository trong nội dung job script. Không cần `conda activate` vì runner đọc đường dẫn Python tuyệt đối từ `.phase2-server-env.sh`.

| Nhóm | Số lượt | Model | GPU đề xuất |
|---|---:|---|---:|
| `vllm` | 8 | Hai Qwen full, DeepSeek Small full, DeepSeek Tiny vá | 2 GPU |
| `deepseek-int8` | 2 | DeepSeek-VL2 8-bit vá | 2 GPU |
| `vintern` | 4 | Vintern 1B và 3B full | 2 GPU |
| `huatuo` | 2 | HuatuoGPT-Vision-34B full | 2 GPU |

Cả bốn allocation đều yêu cầu hai GPU. Cách sử dụng khác nhau theo backend:

- `vllm`: các lượt Qwen dùng tensor parallel trên cả hai GPU; sau đó DeepSeek Small/Tiny chạy bằng hai worker một-GPU.
- Ba nhóm còn lại: runner tạo hai worker, mỗi worker chỉ nhìn thấy một GPU và xử lý các lượt được giao tuần tự.

Runner đọc danh sách thiết bị từ `CUDA_VISIBLE_DEVICES`, kể cả khi scheduler cấp số GPU vật lý hoặc UUID không liên tiếp. Trên server chỉ có hai GPU, submit cả bốn job rồi để scheduler xếp lần lượt; không ép bốn job chạy đồng thời trên cùng hai GPU.

### Job 1: vLLM

```bash
cd /shared/path/DermNet_Dataset
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

Nhóm này dùng `dermnet-vllm` với Torch 2.13.0. Qwen sử dụng toàn bộ GPU mà scheduler cho nhìn thấy, vì vậy job Qwen nên được cấp hai GPU và chỉ nhìn thấy đúng hai GPU được cấp qua `CUDA_VISIBLE_DEVICES`.

Runner chạy tuần tự bốn lượt Qwen bằng cả hai GPU. Khi Qwen hoàn tất, bốn lượt DeepSeek Small/Tiny được chia sang hai worker để cả hai GPU tiếp tục hoạt động.

### Job 2: DeepSeek INT8

```bash
cd /shared/path/DermNet_Dataset
bash Phase_2/VLMEvalKit/run_phase2.sh run-group deepseek-int8
```

Val và Test được chia cho hai GPU và chạy đồng thời.

### Job 3: Vintern

```bash
cd /shared/path/DermNet_Dataset
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern
```

Runner duy trì tối đa hai lượt đồng thời, mỗi GPU một lượt. Khi một worker hoàn tất lượt đầu, nó tiếp tục lượt Vintern còn lại được giao.

### Job 4: Huatuo

```bash
cd /shared/path/DermNet_Dataset
bash Phase_2/VLMEvalKit/run_phase2.sh run-group huatuo
```

Val và Test được chia cho hai GPU và chạy đồng thời. Mỗi GPU 96 GB chứa một bản HuatuoGPT-Vision-34B.

Các dòng `#SBATCH`, PBS hoặc LSF phụ thuộc cấu hình cụm máy nên đặt ở phần đầu job script theo mẫu của quản trị viên. Mỗi job cần yêu cầu hai GPU, ví dụ với Slurm thường có dòng `#SBATCH --gres=gpu:2`; tên resource chính xác vẫn theo quy định của cụm máy. Phần lệnh inference giữ nguyên như trên.

## 3. Chạy thử kế hoạch trước khi submit

Các lệnh dưới đây không cài package và không load model:

```bash
DRY_RUN=1 GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 \
  bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm

DRY_RUN=1 GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 \
  bash Phase_2/VLMEvalKit/run_phase2.sh run-group deepseek-int8

DRY_RUN=1 GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 \
  bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern

DRY_RUN=1 GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 \
  bash Phase_2/VLMEvalKit/run_phase2.sh run-group huatuo
```

Kết quả phải lần lượt có 8, 2, 4 và 2 lượt; tổng cộng đúng 16 lượt.

## 4. Chạy tiếp sau gián đoạn

Submit lại đúng lệnh của nhóm bị dừng. Ví dụ:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

`run-group` tự kiểm tra kết quả đã hoàn chỉnh, bỏ qua phần đã xong và dùng checkpoint cho phần thiếu hoặc thất bại. Không cần đổi sang lệnh khác.

Mỗi nhóm có lock và bộ TSV runtime riêng nên có thể nằm trong các allocation scheduler khác nhau. Không chạy `all`, `resume` hoặc `server` đồng thời với bốn `run-group`, vì các lệnh toàn bộ có thể chọn lại cùng model/output.

## 5. Output và log

Kết quả model vẫn nằm dưới:

```text
Phase_2/VLMEvalKit/outputs/answer-format-v4-vllm/
```

Log và trạng thái nằm dưới:

```text
Phase_2/VLMEvalKit/outputs/answer-format-v4-vllm/.phase2-runner/
```

Các nhóm ghi model khác nhau nên không ghi đè kết quả của nhau. Những lượt full chạy bằng worker một-GPU được tách theo từng model/dataset tại:

```text
Phase_2/VLMEvalKit/outputs/answer-format-v4-vllm/two-gpu-jobs/
```

Nếu cần chẩn đoán bằng một worker nhưng vẫn giữ nguyên lệnh nhóm:

```bash
RUN_GROUP_WORKERS=1 bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern
```

Scheduler vẫn phải cấp đúng hai GPU riêng cho job. Runner chỉ chia hai GPU bên trong một `run-group`, không chia GPU giữa nhiều allocation.
