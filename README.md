# DermNet Dataset

## Cách chạy chính: bốn job trên HPC scheduler

Toàn bộ benchmark được điều khiển bằng đúng một file: `Phase_2/VLMEvalKit/run_phase2.sh`. Environment được chuẩn bị một lần trước khi submit; mỗi compute job sau đó chỉ chạy inference cho một nhóm model.

```text
Login/setup node:  clone code -> tạo 4 environment -> tạo file mapping
Compute nodes:     submit 4 run-group -> inference 16 lượt
Nếu bị gián đoạn:  submit lại đúng run-group bị dừng
```

Hướng dẫn đầy đủ: [chạy bốn job scheduler](docs/SCHEDULER_RUN.md). Chi tiết từng lệnh Conda/pip: [setup server](docs/SERVER_SETUP.md#3-setup-thủ-công-từng-environment).

### Bước 1: clone repository trên filesystem dùng chung

```bash
git clone https://github.com/MinhKemm/DermNet_Dataset.git
cd DermNet_Dataset
```

Repository, environment, thư mục `vendor`, dữ liệu và output cần nằm ở vị trí mà compute node đọc được.

### Bước 2: chuẩn bị environment trước khi submit job

Cách ngắn nhất trên setup node có quyền cài package **và nhìn thấy GPU**:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh setup
```

Nếu quản trị viên muốn cài riêng từng environment, dùng đúng bốn requirement sau; không gộp chúng vào cùng một environment:

| Environment | Requirement | Nhóm chạy |
|---|---|---|
| `dermnet-vllm` | [vllm-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/vllm-blackwell.txt) | Qwen, DeepSeek Small/Tiny |
| `dermnet-deepseek-int8` | [deepseek-int8-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/deepseek-int8-blackwell.txt) | DeepSeek-VL2 8-bit |
| `dermnet-vintern` | [vintern-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/vintern-blackwell.txt) | Vintern 1B/3B |
| `dermnet-huatuo` | [huatuo-blackwell.txt](Phase_2/VLMEvalKit/requirements/server/huatuo-blackwell.txt) | HuatuoGPT-Vision-34B |

Các lệnh cài package cho bốn environment:

```bash
export DERMNET_KIT_DIR="$PWD/Phase_2/VLMEvalKit"
export LEGACY_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

conda create -n dermnet-vllm python=3.10 pip -y
conda run -n dermnet-vllm python -m pip install --upgrade pip setuptools wheel packaging
conda run -n dermnet-vllm python -m pip install \
  -r "$DERMNET_KIT_DIR/requirements/server/vllm-blackwell.txt"
conda run -n dermnet-vllm python -m pip install --no-deps -e "$DERMNET_KIT_DIR"

conda create -n dermnet-deepseek-int8 python=3.10 pip -y
conda run -n dermnet-deepseek-int8 python -m pip install --upgrade pip setuptools wheel packaging
conda run -n dermnet-deepseek-int8 python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url "$LEGACY_TORCH_INDEX_URL"
conda run -n dermnet-deepseek-int8 python -m pip install \
  -r "$DERMNET_KIT_DIR/requirements/server/deepseek-int8-blackwell.txt"
conda run -n dermnet-deepseek-int8 python -m pip install --no-deps -e "$DERMNET_KIT_DIR"

conda create -n dermnet-vintern python=3.10 pip -y
conda run -n dermnet-vintern python -m pip install --upgrade pip setuptools wheel packaging
conda run -n dermnet-vintern python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url "$LEGACY_TORCH_INDEX_URL"
conda run -n dermnet-vintern python -m pip install \
  -r "$DERMNET_KIT_DIR/requirements/server/vintern-blackwell.txt"
conda run -n dermnet-vintern python -m pip install --no-deps -e "$DERMNET_KIT_DIR"

conda create -n dermnet-huatuo python=3.10 pip -y
conda run -n dermnet-huatuo python -m pip install --upgrade pip setuptools wheel packaging
conda run -n dermnet-huatuo python -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url "$LEGACY_TORCH_INDEX_URL"
conda run -n dermnet-huatuo python -m pip install \
  -r "$DERMNET_KIT_DIR/requirements/server/huatuo-blackwell.txt"
conda run -n dermnet-huatuo python -m pip install --no-deps -e "$DERMNET_KIT_DIR"

for DERMNET_ENV in dermnet-vllm dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo; do
  conda run -n "$DERMNET_ENV" python -m pip check
done
```

Sau khi cài package, tiếp tục phần clone source DeepSeek/Huatuo, áp dụng bản vá Blackwell và tạo file mapping theo [setup thủ công đầy đủ](docs/SERVER_SETUP.md#3-setup-thủ-công-từng-environment). Đây là các bước bắt buộc để runner tìm đúng source và đúng Python.

Nếu login node không nhìn thấy GPU, cài thủ công như trên; mỗi compute job sẽ tự kiểm tra CUDA của nhóm trước inference. Sau bước setup đầy đủ phải có file:

```text
Phase_2/VLMEvalKit/.phase2-server-env.sh
```

File này lưu đường dẫn tuyệt đối tới bốn Python environment. Compute job tự đọc file nên không cần `conda activate`. Nếu file mapping nằm ở nơi dùng chung khác, đặt `SERVER_ENV_FILE=/shared/path/.phase2-server-env.sh` trong job.

### Bước 3: submit bốn nhóm inference

Mỗi job scheduler chỉ cần `cd` vào root repository rồi gọi một lệnh dưới đây:

```bash
# Job 1: 2 GPU - Qwen và DeepSeek Small/Tiny
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm

# Job 2: 2 GPU - hai lượt DeepSeek-VL2 8-bit chạy song song
bash Phase_2/VLMEvalKit/run_phase2.sh run-group deepseek-int8

# Job 3: 2 GPU - hai lượt Vintern chạy song song
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern

# Job 4: 2 GPU - hai lượt Huatuo chạy song song
bash Phase_2/VLMEvalKit/run_phase2.sh run-group huatuo
```

| Nhóm | Lượt | Model | GPU đề xuất |
|---|---:|---|---:|
| `vllm` | 8 | Qwen3.5, Qwen3-VL, DeepSeek Small, DeepSeek Tiny | 2 |
| `deepseek-int8` | 2 | DeepSeek-VL2 8-bit | 2 |
| `vintern` | 4 | Vintern 1B và 3B | 2 |
| `huatuo` | 2 | HuatuoGPT-Vision-34B | 2 |

Các dòng `#SBATCH`, PBS hoặc LSF đặt phía trên theo mẫu của cụm máy và yêu cầu **2 GPU cho mỗi job**. Trong nhóm `vllm`, các lượt Qwen dùng đồng thời cả hai GPU; sau khi Qwen xong, DeepSeek Small/Tiny được chia thành hai worker, mỗi GPU một lượt. Ba nhóm còn lại cũng tự tạo hai worker song song. Trên server chỉ có hai GPU, có thể submit cả bốn job rồi để scheduler xếp chúng chạy lần lượt.

Runner giữ nguyên `CUDA_VISIBLE_DEVICES` do scheduler cấp. Kết quả full chạy song song được tách theo từng model/dataset trong `outputs/answer-format-v4-vllm/two-gpu-jobs/`, tránh hai process ghi chung một trạng thái. Nếu cần chẩn đoán với một worker, đặt `RUN_GROUP_WORKERS=1` trước lệnh `run-group`.

### Bước 4: chạy tiếp sau gián đoạn

Submit lại đúng lệnh của nhóm đã dừng. Ví dụ nhóm vLLM bị ngắt:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

Runner bỏ qua kết quả đã hoàn chỉnh và tiếp tục phần thiếu hoặc thất bại. Mỗi nhóm có lock và vùng dữ liệu runtime riêng, vì vậy bốn nhóm có thể nằm trong các allocation khác nhau. Không chạy `all`, `resume` hoặc `server` đồng thời với các `run-group`.

Qwen **bắt buộc dùng vLLM** trong cấu hình hiện tại. Qwen, DeepSeek Small và DeepSeek Tiny dùng environment `dermnet-vllm`; DeepSeek 8-bit dùng Transformers + bitsandbytes; Vintern và Huatuo dùng hai environment Transformers riêng. Chi tiết phiên bản và bản vá Blackwell: [setup server](docs/SERVER_SETUP.md) và [backend Blackwell/vLLM](docs/VLLM_SERVER.md).

## Phương án phụ: máy cho phép setup và inference trong cùng phiên

Trên server không qua scheduler, hoặc phiên shell được phép cài environment rồi chạy inference, có thể gọi toàn bộ tuần tự:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh server
```

Lệnh này thực hiện `setup -> doctor -> all`. Nếu bị gián đoạn sau khi setup hoàn tất:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Danh sách hiện tại có **8 model**, chỉ dùng Val và Test tiếng Việt: **16 lượt = 12 full + 4 vá**. Một lượt là một model chạy trên một bộ dữ liệu. Với `run-group`, runner chạy tối đa hai lượt song song khi backend chỉ cần một GPU; các lượt Qwen dùng cả hai GPU và chạy tuần tự.

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

Các lệnh kiểm tra kế hoạch:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
DRY_RUN=1 bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Bốn lệnh inference riêng, không chạy setup:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
bash Phase_2/VLMEvalKit/run_phase2.sh run-group deepseek-int8
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern
bash Phase_2/VLMEvalKit/run_phase2.sh run-group huatuo
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

## Environment và requirements

`requirements.txt` là dependency lõi của VLMEvalKit, không đại diện cho backend của mọi model. Bốn profile chạy server nằm tại `Phase_2/VLMEvalKit/requirements/server/`; lệnh `setup` cài profile tương ứng:

- `vllm-blackwell.txt`: Qwen3.5, Qwen3-VL, DeepSeek Small/Tiny.
- `deepseek-int8-blackwell.txt`: DeepSeek 8-bit.
- `vintern-blackwell.txt`: Vintern 1B/3B.
- `huatuo-blackwell.txt`: Huatuo 34B.

Conda, Git, driver NVIDIA và `nvidia-smi` cần có trên server.

Setup ghim commit của mã nguồn DeepSeek-VL2 và HuatuoGPT-Vision để lần cài sau không tự đổi code, rồi áp dụng bản vá attention đã kiểm thử đúng vào chính hai source tree này. Khi model gated yêu cầu xác thực tải, đặt `HF_TOKEN` bằng cơ chế secret của server. Cuối lệnh `setup`, doctor được chạy tự động; vẫn có thể gọi lại `doctor` bất cứ lúc nào.

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
python -m unittest discover -s tests -p 'test_*.py'
bash -n run_phase2.sh
```

Đã kiểm tra tương thích 4 nguồn patch; chưa chạy suy luận model trên server GPU. Chi tiết: [rà soát Excel và kế hoạch](docs/RESULTS_PATCH_AUDIT.md).
