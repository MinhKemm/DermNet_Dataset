# Hướng dẫn sửa environment HPC đã cài sẵn

Tài liệu này chỉ là note để người setup copy lệnh. **Không chạy file fix/setup
tự động trong tình trạng hiện tại**: bốn environment đã được cài trên login node,
và server đã chưa được thay đổi trong hai ngày qua.

HPC của server tách hai loại node:

- **Login node:** kiểm tra và thay đổi package trong environment đã có.
- **Run/compute node:** chỉ chạy environment đã có; tuyệt đối không `conda
  create`, `pip install`, nâng Python hoặc chạy file fix.

Không chạy `run_phase2.sh server`, `run_phase2.sh setup`,
`fix_qwen_vllm_env.sh` hoặc `prepare-runtime` để tạo lại environment. Phần
`run_phase2.sh run-group` ở cuối tài liệu vẫn được dùng để chạy benchmark.

## 1. Kiểm tra environment hiện có — chỉ chạy trên login node

Không thay đổi gì ở bước này:

```bash
cd /shared/homes/u26466553/projects/DermNet_Dataset

for ENV in dermnet-vllm dermnet-vllm-py312 dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo
do
  echo "===== $ENV ====="
  conda run --no-capture-output -n "$ENV" python -c '
import importlib.metadata as metadata
import sys

print("python=" + sys.version.split()[0])
print("python_exe=" + sys.executable)
for name in [
    "vllm", "flashinfer-python", "flashinfer-cubin",
    "torch", "torchvision", "transformers", "qwen-vl-utils",
    "bitsandbytes", "accelerate", "timm", "einops", "einops-exts",
    "peft", "sentencepiece", "numpy", "tokenizers",
]:
    try:
        print(name + "=" + metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name + "=MISSING")
' 2>&1 || true
done
```

Lấy đúng `python_exe` của từng environment để đưa vào job script:

```bash
for ENV in dermnet-vllm dermnet-vllm-py312 dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo
do
  printf '%s: ' "$ENV"
  conda run --no-capture-output -n "$ENV" python -c 'import sys; print(sys.executable)' \
    | awk 'NF {line=$0} END {print line}'
done
```

Kiểm tra dependency conflict, vẫn không thay đổi package:

```bash
for ENV in dermnet-vllm dermnet-vllm-py312 dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo
do
  echo "===== pip check: $ENV ====="
  conda run -n "$ENV" python -m pip check 2>&1 || true
done
```

Nếu `dermnet-vllm` là Python 3.10 nhưng `dermnet-vllm-py312` là Python 3.12,
dùng `dermnet-vllm-py312` cho Qwen. Không nâng trực tiếp hoặc xóa
`dermnet-vllm` cũ.

## 2. Version cần có

Các version dưới đây là mục tiêu để đối chiếu với kết quả Bước 1. Những package
ghi **chưa pin** không cần tự đổi version nếu `pip check` không báo lỗi.

| Environment/model | Python | Package chính cần có |
|---|---:|---|
| `dermnet-vllm` hoặc env vLLM Python 3.12 — Qwen, DeepSeek Small/Tiny | 3.12 | `vllm==0.28.0`, `flashinfer-python==0.6.16.post3`, `flashinfer-cubin==0.6.16.post3`, `torch==2.13.0`, `torchvision==0.28.0`, `transformers==5.17.0`, `qwen-vl-utils==0.0.14` |
| `dermnet-deepseek-int8` — DeepSeek-VL2 INT8 | 3.10 | `torch==2.8.0`, `torchvision==0.23.0`, `transformers==4.38.2`, `bitsandbytes==0.49.0`; `attrdict`, `einops`, `timm`, `sentencepiece` — chưa pin |
| `dermnet-vintern` — Vintern 1B/3B | 3.10 | `torch==2.8.0`, `torchvision==0.23.0`, `transformers==4.42.3`; `einops`, `timm`, `sentencepiece` — chưa pin |
| `dermnet-huatuo` — HuatuoGPT-Vision | 3.10 | `torch==2.8.0`, `torchvision==0.23.0`, `transformers==4.37.2`, `tokenizers==0.15.2`, `numpy==1.26.4`, `accelerate==0.21.0`, `bitsandbytes==0.49.0`, `einops==0.6.1`, `einops-exts==0.0.4`, `peft==0.4.0`, `sentencepiece==0.1.99`, `shortuuid==1.0.11`, `timm==0.6.13`, `markdown2==2.4.10`, `wavedrom==2.0.3.post3` |

Torch của ba environment legacy lấy từ CUDA 12.8 index khi cần sửa:

```text
https://download.pytorch.org/whl/cu128
```

Không cài FlashInfer vào ba environment legacy.

## 3. Sửa đúng lỗi Qwen — chỉ trên login node

Log `dermnet_vllm.o85842` cho thấy Qwen chết trước inference tại
`flashinfer/comm/fd_exchange.py`:

```text
TypeError: 'type' object is not subscriptable
array.array[int]
```

Nguyên nhân là vLLM đang chạy bằng Python 3.10, trong khi FlashInfer 0.6.16
đang dùng annotation cần Python 3.12. Vì vậy chỉ xóa/cài lại FlashInfer trong
Python 3.10 **không đủ để sửa**.

### Trường hợp A — đã có env Python 3.12

Đặt tên đúng env 3.12 đã thấy ở Bước 1:

```bash
QWEN_ENV=dermnet-vllm-py312
```

Nếu env 3.12 thực tế tên là `dermnet-vllm`, đổi lại giá trị trên. Kiểm tra:

```bash
conda run -n "$QWEN_ENV" python -c \
  'import sys; print(sys.version); print(sys.executable)'
```

Nếu FlashInfer chưa đúng version, chỉ thay hai package này, không đụng các
environment legacy:

```bash
conda run -n "$QWEN_ENV" python -m pip install \
  --upgrade --force-reinstall --no-deps \
  --extra-index-url https://flashinfer.ai/whl/ \
  flashinfer-python==0.6.16.post3 \
  flashinfer-cubin==0.6.16.post3
```

Nếu `vllm` cũng sai version, sửa riêng nó:

```bash
conda run -n "$QWEN_ENV" python -m pip install \
  --upgrade --no-deps vllm==0.28.0
```

### Trường hợp B — chỉ có env vLLM Python 3.10

Không cài FlashInfer 0.6.16 vào env Python 3.10 với kỳ vọng lỗi sẽ hết. Dùng
một env Python 3.12 đã tồn tại nếu server có sẵn. Chỉ khi server **thật sự
không có** env Python 3.12 mới tạo thêm một env trên login node; không tạo ở
run node:

```bash
conda create -n dermnet-vllm-py312 python=3.12 pip -y
conda run -n dermnet-vllm-py312 python -m pip install -U pip setuptools wheel packaging
conda run -n dermnet-vllm-py312 python -m pip install \
  --extra-index-url https://flashinfer.ai/whl/ \
  vllm==0.28.0 \
  flashinfer-python==0.6.16.post3 \
  flashinfer-cubin==0.6.16.post3 \
  torch==2.13.0 torchvision==0.28.0 \
  transformers==5.17.0 qwen-vl-utils==0.0.14
```

Đây là phương án dự phòng vì environment hiện tại được báo là đã cài sẵn,
không phải bước mặc định.

## 4. Kiểm tra Qwen và lỗi `LD_LIBRARY_PATH` — vẫn trên login node

```bash
# Giữ đúng tên env Python 3.12 đã chọn ở Bước 3.
QWEN_ENV=dermnet-vllm-py312
QWEN_PYTHON="$(conda run --no-capture-output -n "$QWEN_ENV" python -c 'import sys; print(sys.executable)' | awk 'NF {line=$0} END {print line}')"

env -u LD_LIBRARY_PATH "$QWEN_PYTHON" -c '
import importlib.metadata as metadata
import os
from array import array

assert "LD_LIBRARY_PATH" not in os.environ
assert array[int]
assert metadata.version("vllm") == "0.28.0"
assert metadata.version("flashinfer-python") == "0.6.16.post3"
assert metadata.version("flashinfer-cubin") == "0.6.16.post3"
import flashinfer.comm
import vllm.distributed.device_communicators.flashinfer_all_reduce
print("Qwen environment: OK")
'

"$QWEN_PYTHON" -m pip check
```

`env -u LD_LIBRARY_PATH` chỉ bỏ biến này khỏi process được chạy; nó không sửa
hay xóa library của hệ thống. Khi chạy vLLM trên compute node cũng phải giữ
đúng workaround này.

## 5. Chuẩn bị job compute bằng path đã có — không cài gì trong job

Trong job script, điền các path tuyệt đối đã lấy ở Bước 1. Ví dụ:

```bash
QWEN_PYTHON=/shared/homes/u26466553/miniconda3/envs/dermnet-vllm-py312/bin/python
DEEPSEEK_PYTHON=/shared/homes/u26466553/miniconda3/envs/dermnet-deepseek-int8/bin/python
VINTERN_PYTHON=/shared/homes/u26466553/miniconda3/envs/dermnet-vintern/bin/python
HUATUO_PYTHON=/shared/homes/u26466553/miniconda3/envs/dermnet-huatuo/bin/python

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
```

Các path trên chỉ là ví dụ; phải thay bằng `python_exe` thật ở Bước 1. Job
không được gọi `conda create`, `pip install`, `setup` hay file fix.

Kiểm tra GPU và toàn bộ dependency trên compute node:

```bash
env -u LD_LIBRARY_PATH \
PYTHON_BIN="$QWEN_PYTHON" \
PYTHON_QWEN="$QWEN_PYTHON" \
PYTHON_DEEPSEEK_VLLM="$QWEN_PYTHON" \
PYTHON_DEEPSEEK="$DEEPSEEK_PYTHON" \
PYTHON_VINTERN="$VINTERN_PYTHON" \
PYTHON_HUATUO="$HUATUO_PYTHON" \
SERVER_ENV_FILE=/dev/null \
  bash run_phase2.sh doctor
```

## 6. Chạy model tuần tự

Chỉ submit group tiếp theo sau khi group trước kết thúc thành công. Biến
`RUN_GROUP_WORKERS=1` buộc các lượt trong một group chạy lần lượt, không chạy
đồng thời.

### Job 1 — vLLM: Qwen và DeepSeek Small/Tiny

Đây là job phải dùng `env -u LD_LIBRARY_PATH`:

```bash
env -u LD_LIBRARY_PATH \
  SERVER_ENV_FILE=/dev/null \
  PYTHON_BIN="$QWEN_PYTHON" \
  PYTHON_QWEN="$QWEN_PYTHON" \
  PYTHON_DEEPSEEK_VLLM="$QWEN_PYTHON" \
  RUN_GROUP_WORKERS=1 \
  bash run_phase2.sh run-group vllm
```

### Job 2 — DeepSeek INT8

Chỉ submit sau khi Job 1 đã xong:

```bash
PYTHON_BIN="$DEEPSEEK_PYTHON" \
PYTHON_DEEPSEEK="$DEEPSEEK_PYTHON" \
SERVER_ENV_FILE=/dev/null \
RUN_GROUP_WORKERS=1 \
  bash run_phase2.sh run-group deepseek-int8
```

### Job 3 — Vintern

```bash
PYTHON_BIN="$VINTERN_PYTHON" \
PYTHON_VINTERN="$VINTERN_PYTHON" \
SERVER_ENV_FILE=/dev/null \
RUN_GROUP_WORKERS=1 \
  bash run_phase2.sh run-group vintern
```

### Job 4 — Huatuo

```bash
PYTHON_BIN="$HUATUO_PYTHON" \
PYTHON_HUATUO="$HUATUO_PYTHON" \
SERVER_ENV_FILE=/dev/null \
RUN_GROUP_WORKERS=1 \
  bash run_phase2.sh run-group huatuo
```

Không submit cả bốn job đồng thời trên cùng allocation. Chạy tuần tự giúp
không tạo thêm worker song song, nhưng không thể tự giải phóng GPU đang bị một
process khác chiếm.

## 7. Lỗi memory DeepSeek là lỗi khác

Log cũ ghi:

```text
Free memory: 69.64/94.97 GiB
vLLM desired: khoảng 75.98 GiB
```

Đây là thiếu VRAM trên GPU được cấp, không phải lỗi FlashInfer và không giải
quyết bằng cách cài lại package. Cần xin allocation sạch hoặc nhờ admin xử lý
process khác đang chiếm GPU; không tự kill process của người khác.

## Kết luận ngắn

1. Không tạo lại bốn environment đã có.
2. Kiểm tra version trên login node.
3. Qwen phải chạy bằng env Python 3.12 và đúng cặp FlashInfer.
4. Dùng `env -u LD_LIBRARY_PATH` cho job vLLM.
5. Không cài gì trên run node.
6. Submit bốn group lần lượt với `RUN_GROUP_WORKERS=1`.
