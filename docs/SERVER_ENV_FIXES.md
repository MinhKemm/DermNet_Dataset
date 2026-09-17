# Chạy benchmark HPC khi environment đã cài sẵn

## Kết luận trước

Environment trên server **đã có sẵn**. Vì vậy không chạy lại các lệnh tạo/cài
environment. Hiện repo chưa tự thay đổi server; người setup cần làm đúng các
bước dưới đây trên server.

| Nơi chạy | Được làm | Không được làm |
|---|---|---|
| Login node | Kiểm tra version; nếu Qwen sai thì sửa riêng env Qwen | Không xóa env cũ |
| Run/compute node | Chạy benchmark bằng env đã có | Không `conda create`, `pip install`, nâng Python, hoặc chạy file fix |

Không chạy các lệnh sau trong quy trình hiện tại: `run_phase2.sh server`,
`run_phase2.sh setup`, `prepare-runtime` và `fix_qwen_vllm_env.sh`. Đây là các
lệnh setup/fix tự động, không cần dùng khi env đã được cài sẵn.

## Bước 1 — Kiểm tra env trên login node, chưa thay đổi gì

```bash
cd /shared/homes/u26466553/projects/DermNet_Dataset

for ENV in dermnet-vllm dermnet-vllm-py312 dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo
do
  echo "===== $ENV ====="
  conda run --no-capture-output -n "$ENV" python -c '
import importlib.metadata as m
import sys

print("Python=" + sys.version.split()[0])
print("Python executable=" + sys.executable)
for package in [
    "vllm", "flashinfer-python", "flashinfer-cubin", "torch",
    "torchvision", "transformers", "qwen-vl-utils", "bitsandbytes",
]:
    try:
        print(package + "=" + m.version(package))
    except m.PackageNotFoundError:
        print(package + "=MISSING")
' 2>&1 || true
done
```

Ghi lại đường dẫn `Python executable` của các env. Dùng đúng đường dẫn đó trong
job script, không cần `conda activate`:

```bash
for ENV in dermnet-vllm dermnet-vllm-py312 dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo
do
  printf '%s: ' "$ENV"
  conda run --no-capture-output -n "$ENV" python -c 'import sys; print(sys.executable)' \
    | awk 'NF {line=$0} END {print line}'
done
```

## Bước 2 — Đọc kết quả Qwen

Qwen chỉ đạt khi env mà Qwen sử dụng có đủ cả ba điều kiện:

```text
Python                 3.12
vllm                   0.28.0
flashinfer-python      0.6.16.post3
flashinfer-cubin       0.6.16.post3
```

Nếu `dermnet-vllm` là Python 3.10 nhưng `dermnet-vllm-py312` là Python 3.12,
chọn `dermnet-vllm-py312` cho Qwen. Không sửa hoặc xóa `dermnet-vllm` cũ.

Lỗi Qwen trong log là:

```text
TypeError: 'type' object is not subscriptable
array.array[int]
```

Lỗi này do FlashInfer chạy bằng Python 3.10. Cài lại riêng FlashInfer trong
Python 3.10 không giải quyết được; phải chạy Qwen bằng env Python 3.12.

Đọc kết quả theo bảng này:

| Kết quả kiểm tra | Việc cần làm |
|---|---|
| Python 3.12 và cả ba version đúng | Không cài gì; chuyển sang Bước 3 |
| Python 3.12 nhưng FlashInfer sai | Cài lại đúng hai package FlashInfer ở lệnh bên dưới |
| Qwen chỉ có Python 3.10 | Không chạy Qwen bằng env đó; chọn env Python 3.12 đã có |
| Ba env legacy sai package | Không sửa trong phạm vi lỗi Qwen này |

Nếu env Python 3.12 đã có nhưng FlashInfer sai version, chỉ chạy hai lệnh sau
trên **login node**. Đổi `QWEN_ENV` thành đúng tên env ở Bước 1:

```bash
QWEN_ENV=dermnet-vllm-py312

conda run -n "$QWEN_ENV" python -m pip install \
  --upgrade --force-reinstall --no-deps \
  --extra-index-url https://flashinfer.ai/whl/ \
  flashinfer-python==0.6.16.post3 \
  flashinfer-cubin==0.6.16.post3

conda run -n "$QWEN_ENV" python -m pip check
```

Nếu `vllm` sai version, sửa thêm riêng trong chính env Qwen:

```bash
conda run -n "$QWEN_ENV" python -m pip install --upgrade --no-deps vllm==0.28.0
```

Ba env còn lại (`dermnet-deepseek-int8`, `dermnet-vintern`,
`dermnet-huatuo`) không cần Python 3.12 và không cài FlashInfer vào đó.

Nếu Bước 1 chứng minh server **không có bất kỳ env Python 3.12 nào**, dừng
trước khi chạy Qwen và báo người quản trị. Chỉ khi được phép tạo thêm env trên
login node mới dùng cách dự phòng sau; tuyệt đối không chạy nó trong job:

```bash
conda create -n dermnet-vllm-py312 python=3.12 pip -y
conda run -n dermnet-vllm-py312 python -m pip install \
  -r Phase_2/VLMEvalKit/requirements/server/vllm-blackwell.txt
conda run -n dermnet-vllm-py312 python -m pip install --no-deps \
  -e Phase_2/VLMEvalKit
```

Sau đó chạy lại Bước 1 và Bước 2; không tự đoán path Python.

## Bước 3 — Tắt lỗi `LD_LIBRARY_PATH` trong job vLLM

Không cần gõ `env -u LD_LIBRARY_PATH` trước từng lệnh. Thêm đúng một dòng ở
đầu **job script chạy vLLM trên compute node**:

```bash
unset LD_LIBRARY_PATH
```

Dòng này chỉ có hiệu lực trong job hiện tại và các process con của job. Không
thêm vào `.bashrc` dùng chung vì có thể ảnh hưởng env legacy.

## Bước 4 — Chạy từng group, không chạy đồng thời

Mỗi lệnh dưới đây là **một job scheduler riêng**. Các đường dẫn chỉ là ví dụ;
thay bằng `Python executable` thật lấy ở Bước 1.

### Job 1: Qwen + DeepSeek Small/Tiny

Đây là job vLLM nên có `unset LD_LIBRARY_PATH`:

```bash
#!/bin/bash
unset LD_LIBRARY_PATH
export SERVER_ENV_FILE=/dev/null
export PYTHON_BIN=/shared/homes/u26466553/miniconda3/envs/dermnet-vllm-py312/bin/python
export PYTHON_QWEN="$PYTHON_BIN"
export PYTHON_DEEPSEEK_VLLM="$PYTHON_BIN"
export RUN_GROUP_WORKERS=1

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
bash run_phase2.sh run-group vllm
```

`SERVER_ENV_FILE=/dev/null` chỉ bảo runner không đọc mapping cũ; không xóa file
nào. Nếu mapping hiện tại đã đúng path Python, có thể bỏ dòng này.

### Job 2: DeepSeek INT8

Chỉ submit sau khi Job 1 kết thúc:

```bash
#!/bin/bash
export SERVER_ENV_FILE=/dev/null
export PYTHON_BIN=/shared/homes/u26466553/miniconda3/envs/dermnet-deepseek-int8/bin/python
export PYTHON_DEEPSEEK="$PYTHON_BIN"
export RUN_GROUP_WORKERS=1

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
bash run_phase2.sh run-group deepseek-int8
```

### Job 3: Vintern

```bash
#!/bin/bash
export SERVER_ENV_FILE=/dev/null
export PYTHON_BIN=/shared/homes/u26466553/miniconda3/envs/dermnet-vintern/bin/python
export PYTHON_VINTERN="$PYTHON_BIN"
export RUN_GROUP_WORKERS=1

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
bash run_phase2.sh run-group vintern
```

### Job 4: Huatuo

```bash
#!/bin/bash
export SERVER_ENV_FILE=/dev/null
export PYTHON_BIN=/shared/homes/u26466553/miniconda3/envs/dermnet-huatuo/bin/python
export PYTHON_HUATUO="$PYTHON_BIN"
export HUATUO_SOURCE_DIR=/shared/homes/u26466553/projects/DermNet_Dataset/vendor/HuatuoGPT-Vision
export RUN_GROUP_WORKERS=1

cd /shared/homes/u26466553/projects/DermNet_Dataset/Phase_2/VLMEvalKit
bash run_phase2.sh run-group huatuo
```

Thứ tự submit là: Job 1 xong → Job 2 xong → Job 3 xong → Job 4. Nếu một job
dừng giữa chừng, submit lại đúng job đó; runner sẽ dùng checkpoint và bỏ qua
phần đã hoàn tất.

## Bước 5 — Nếu còn lỗi memory

Log cũ của DeepSeek báo:

```text
GPU còn: 69.64/94.97 GiB
vLLM cần: khoảng 75.98 GiB
```

Đây là thiếu VRAM, không phải lỗi FlashInfer. Không cài lại package và không tự
kill process của người khác. Cần xin GPU allocation sạch hoặc nhờ admin xử lý
process đang chiếm GPU.

## Tóm tắt một dòng

Env đã có → kiểm tra Qwen phải là Python 3.12 + đúng FlashInfer → đặt
`unset LD_LIBRARY_PATH` ở đầu job vLLM → submit 4 job theo đúng thứ tự, mỗi job
dùng `RUN_GROUP_WORKERS=1`.
