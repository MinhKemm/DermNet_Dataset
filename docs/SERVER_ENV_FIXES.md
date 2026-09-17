# Xử lý server HPC — environment đã cài sẵn

## Cần làm gì?

Hiện bốn environment đã được cài trên **login node**. Repo chưa tự thay đổi
server trong hai ngày qua. Người setup chỉ cần:

1. Kiểm tra env Qwen đang dùng Python và FlashInfer phiên bản nào.
2. Nếu Qwen sai, sửa **riêng env Qwen trên login node**.
3. Trong job vLLM, chạy `unset LD_LIBRARY_PATH` một lần ở đầu job.
4. Submit từng group model, group trước xong mới submit group sau.

Không cần tạo lại bốn environment.

## Quy tắc node

| Node | Được làm | Không được làm |
|---|---|---|
| Login node | `conda run`, kiểm tra version, sửa package Qwen nếu thật sự sai | Không xóa env cũ |
| Run/compute node | Chạy benchmark bằng env đã có | Không `conda create`, `pip install`, nâng Python hoặc chạy file setup/fix |

Không dùng các lệnh setup sau trong quy trình này:
`run_phase2.sh server`, `run_phase2.sh setup`, `prepare-runtime` và
`fix_qwen_vllm_env.sh`.

## Bước 1 — Kiểm tra, không thay đổi gì

Chạy nguyên khối này trên **login node**. Đây là lệnh chỉ đọc:

```bash
set -u
cd /shared/homes/u26466553/projects/DermNet_Dataset
command -v conda >/dev/null || { echo 'ERROR: conda chưa ở PATH'; exit 1; }

ENVS=(dermnet-vllm dermnet-vllm-py312 dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo)
for ENV in "${ENVS[@]}"; do
  echo "===== $ENV ====="
  if conda run --no-capture-output -n "$ENV" python -c '
import importlib.metadata as m
import sys
print("Python=" + sys.version.split()[0])
print("Executable=" + sys.executable)
for p in ["vllm", "flashinfer-python", "flashinfer-cubin", "torch", "transformers", "bitsandbytes"]:
    try:
        print(p + "=" + m.version(p))
    except m.PackageNotFoundError:
        print(p + "=MISSING")
'; then
    :
  else
    echo "STATUS=ENV_NOT_FOUND_OR_IMPORT_FAILED"
  fi
done
```

Lấy đúng dòng `Executable` của env cần dùng. Không tự đoán đường dẫn.

## Bước 2 — Quyết định có phải sửa Qwen không

Env chạy Qwen phải có đúng các giá trị sau:

| Thành phần | Giá trị bắt buộc |
|---|---:|
| Python | `3.12` |
| `vllm` | `0.28.0` |
| `flashinfer-python` | `0.6.16.post3` |
| `flashinfer-cubin` | `0.6.16.post3` |

Đọc kết quả như sau:

- Đủ cả bốn giá trị: **không cài gì**, chuyển sang Bước 3.
- Python 3.12 nhưng FlashInfer sai: chỉ sửa hai package FlashInfer ở Bước 2A.
- Qwen đang dùng Python 3.10: không cài lại FlashInfer vào Python 3.10; chọn
  env Python 3.12 đã có.
- Ba env legacy sai package: không sửa trong phạm vi lỗi Qwen.

### Bước 2A — Chỉ sửa FlashInfer nếu version sai

Chạy trên **login node**. Thay giá trị `QWEN_ENV` bằng đúng env Python 3.12
đã chọn ở Bước 1. Khối lệnh kiểm tra đúng env trước khi cài và dừng nếu sai:

```bash
set -Eeuo pipefail
QWEN_ENV=dermnet-vllm-py312

command -v conda >/dev/null || { echo 'ERROR: conda chưa ở PATH'; exit 1; }
if [[ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ]]; then
  echo 'ERROR: đang ở scheduler job; lệnh sửa chỉ chạy trên login node'
  exit 1
fi
conda run -n "$QWEN_ENV" python -c 'import sys; print(sys.executable)' >/dev/null \
  || { echo "ERROR: không tìm thấy env $QWEN_ENV"; exit 1; }
QWEN_PYTHON="$(conda run --no-capture-output -n "$QWEN_ENV" python -c 'import sys; print(sys.executable)' | awk 'NF {line=$0} END {print line}')"
test -x "$QWEN_PYTHON" || { echo "ERROR: không tìm thấy $QWEN_PYTHON"; exit 1; }

"$QWEN_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

"$QWEN_PYTHON" -m pip install \
  --upgrade --force-reinstall --no-deps \
  --extra-index-url https://flashinfer.ai/whl/ \
  flashinfer-python==0.6.16.post3 \
  flashinfer-cubin==0.6.16.post3

"$QWEN_PYTHON" -m pip check
```

Chỉ khi `vllm` cũng không phải `0.28.0` mới chạy thêm:

```bash
"$QWEN_PYTHON" -m pip install --upgrade --no-deps vllm==0.28.0
"$QWEN_PYTHON" -m pip check
```

Nếu Bước 1 không có env Python 3.12, **dừng và báo người quản trị**. Không tự
tạo env trên run node; việc tạo env mới là quy trình khác.

Lỗi Qwen trong log là:

```text
TypeError: 'type' object is not subscriptable
array.array[int]
```

Nguyên nhân là FlashInfer chạy bằng Python 3.10. Cài lại riêng FlashInfer trong
Python 3.10 không sửa được lỗi này.

## Bước 3 — Kiểm tra Qwen sau khi chọn/sửa env

Đổi `QWEN_ENV` cho đúng tên env. Khối này không cài package:

```bash
set -Eeuo pipefail
QWEN_ENV=dermnet-vllm-py312
QWEN_PYTHON="$(conda run --no-capture-output -n "$QWEN_ENV" python -c 'import sys; print(sys.executable)' | awk 'NF {line=$0} END {print line}')"
test -x "$QWEN_PYTHON"

env -u LD_LIBRARY_PATH "$QWEN_PYTHON" -c '
import importlib.metadata as m
from array import array
assert array[int]
assert m.version("vllm") == "0.28.0"
assert m.version("flashinfer-python") == "0.6.16.post3"
assert m.version("flashinfer-cubin") == "0.6.16.post3"
import flashinfer.comm
import vllm.distributed.device_communicators.flashinfer_all_reduce
print("QWEN_ENV_OK")
'
```

Không thấy `QWEN_ENV_OK` thì chưa submit Job 1.

## Bước 4 — Đầu job vLLM: bỏ `LD_LIBRARY_PATH`

Trong file submit/job script của **Job 1**, đặt sau dòng `#!/bin/bash`:

```bash
set -Eeuo pipefail
unset LD_LIBRARY_PATH
```

`unset` chỉ có hiệu lực trong job hiện tại và process con. Không thêm vào
`.bashrc` dùng chung. Các job legacy không cần đổi `LD_LIBRARY_PATH` theo lỗi
đang có.

## Bước 5 — Submit từng group, không chạy đồng thời

Mỗi khối dưới đây là **một job scheduler riêng**. Các path `/path/...` phải
được thay bằng path `Executable` thật ở Bước 1. `SERVER_ENV_FILE=/dev/null`
giúp dùng trực tiếp path đã ghi trong job thay vì mapping cũ; nó không xóa file.

### Job 1 — vLLM: Qwen + DeepSeek Small/Tiny

```bash
set -Eeuo pipefail
unset LD_LIBRARY_PATH
export SERVER_ENV_FILE=/dev/null
export PYTHON_BIN=/path/to/qwen-python-3.12/bin/python
export PYTHON_QWEN="$PYTHON_BIN"
export PYTHON_DEEPSEEK_VLLM="$PYTHON_BIN"
export RUN_GROUP_WORKERS=1

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
test -x "$PYTHON_BIN"
bash run_phase2.sh run-group vllm
```

### Job 2 — DeepSeek INT8

Chỉ submit sau khi Job 1 kết thúc:

```bash
set -Eeuo pipefail
export SERVER_ENV_FILE=/dev/null
export PYTHON_BIN=/path/to/deepseek-int8/bin/python
export PYTHON_DEEPSEEK="$PYTHON_BIN"
export RUN_GROUP_WORKERS=1

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
test -x "$PYTHON_BIN"
bash run_phase2.sh run-group deepseek-int8
```

### Job 3 — Vintern

```bash
set -Eeuo pipefail
export SERVER_ENV_FILE=/dev/null
export PYTHON_BIN=/path/to/vintern/bin/python
export PYTHON_VINTERN="$PYTHON_BIN"
export RUN_GROUP_WORKERS=1

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
test -x "$PYTHON_BIN"
bash run_phase2.sh run-group vintern
```

### Job 4 — Huatuo

```bash
set -Eeuo pipefail
export SERVER_ENV_FILE=/dev/null
export PYTHON_BIN=/path/to/huatuo/bin/python
export PYTHON_HUATUO="$PYTHON_BIN"
export HUATUO_SOURCE_DIR=/path/to/vendor/HuatuoGPT-Vision
export RUN_GROUP_WORKERS=1

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
test -x "$PYTHON_BIN"
test -f "$HUATUO_SOURCE_DIR/cli.py"
bash run_phase2.sh run-group huatuo
```

Thứ tự bắt buộc: **Job 1 xong → Job 2 xong → Job 3 xong → Job 4**. Không submit
cả bốn job cùng lúc. `RUN_GROUP_WORKERS=1` đảm bảo các lượt trong từng group
cũng chạy lần lượt.

## Bước 6 — Nếu còn lỗi memory

Log cũ của DeepSeek:

```text
GPU còn: 69.64/94.97 GiB
vLLM cần: khoảng 75.98 GiB
```

Đây là thiếu VRAM, không phải lỗi FlashInfer. Không cài lại package và không tự
kill process của người khác; cần allocation GPU sạch hoặc nhờ admin xử lý.

## Tóm tắt

`env đã có` → `Qwen dùng Python 3.12 + FlashInfer đúng version` → `unset
LD_LIBRARY_PATH` ở đầu Job 1 → chạy 4 job theo đúng thứ tự với
`RUN_GROUP_WORKERS=1`.
