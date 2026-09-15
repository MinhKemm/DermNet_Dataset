# Hướng dẫn sửa environment HPC theo đúng thứ tự

Tài liệu này dành cho người trực tiếp setup server. **Code/docs đã được push
nhưng chưa tự sửa environment trên HPC**; theo hiện trạng được báo, server chưa
được sửa trong hai ngày qua. Phải chạy các bước bên dưới trên login node thì
environment mới thực sự thay đổi.

## Quy trình bắt buộc

Làm đúng thứ tự, không nhảy bước.

### Bước 0 — Phân biệt hai loại node

- **Login/setup node:** được phép `conda create` và `pip install`.
- **Run/compute node:** chỉ dùng environment đã cài sẵn để chạy job; không cài
  package, không tạo environment, không chạy file fix.

Nếu đang ở run node thì thoát job và quay về login node trước khi làm Bước 1.

### Bước 1 — Kiểm tra hiện trạng, không thay đổi gì

Chạy trên login node:

```bash
cd /shared/homes/u26466553/projects/DermNet_Dataset

for ENV in dermnet-vllm dermnet-vllm-py312 dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo
do
  echo "===== $ENV ====="
  conda run -n "$ENV" python --version 2>&1 || true
done
```

Nếu `dermnet-vllm` đang là Python 3.10 thì **không xóa và không nâng trực tiếp**;
tạo environment mới `dermnet-vllm-py312` ở Bước 2.

### Bước 2 — Cài/nâng environment trên login node

Nếu cần cài đủ cả bốn environment, dùng đúng một lệnh:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh setup
```

Lệnh này tạo vLLM bằng Python 3.12, tạo ba environment legacy bằng Python 3.10,
cài đúng các version ở bảng bên dưới, tạo launcher sạch và **không cần GPU**.

Nếu ba environment legacy đã có và chỉ Qwen đang lỗi, dùng file fix:

```bash
bash Phase_2/VLMEvalKit/scripts/fix_qwen_vllm_env.sh
bash Phase_2/VLMEvalKit/run_phase2.sh prepare-runtime
```

Hai lệnh trên cũng chỉ chạy trên login node. Không chạy `run_phase2.sh server`
trong mô hình HPC tách login/run vì lệnh đó còn tiếp tục chạy inference.

### Bước 3 — Kiểm tra version sau khi cài

Vẫn trên login node:

```bash
for ENV in dermnet-vllm-py312 dermnet-deepseek-int8 dermnet-vintern dermnet-huatuo
do
  conda run -n "$ENV" python -m pip check
done
```

Kiểm tra đúng lỗi Qwen và workaround:

```bash
VLLM_PYTHON="$(conda run --no-capture-output -n dermnet-vllm-py312 python -c 'import sys; print(sys.executable)' | awk 'NF {line=$0} END {print line}')"
VLLM_CLEAN="$(dirname "$VLLM_PYTHON")/dermnet-vllm-clean"

test -x "$VLLM_CLEAN"
"$VLLM_CLEAN" -c '
import importlib.metadata as m
from array import array
assert array[int]
assert m.version("vllm") == "0.28.0"
assert m.version("flashinfer-python") == "0.6.16.post3"
assert m.version("flashinfer-cubin") == "0.6.16.post3"
import flashinfer.comm
import vllm.distributed.device_communicators.flashinfer_all_reduce
print("Qwen vLLM environment: OK")
'
```

Nếu Bước 3 chưa pass thì chưa được submit inference.

### Bước 4 — Chạy kiểm tra trên run/compute node

Sau khi Bước 3 pass, submit job compute và chỉ chạy:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh doctor
```

`doctor` mới là nơi kiểm tra CUDA, SM 12.0 và memory của node được cấp. Không
chạy `pip`, `conda`, `setup` hoặc file fix trong job này.

### Bước 5 — Chạy model tuần tự

Nếu yêu cầu mỗi model chạy xong mới chuyển model tiếp theo, dùng:

```bash
RUN_GROUP_WORKERS=1 bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
RUN_GROUP_WORKERS=1 bash Phase_2/VLMEvalKit/run_phase2.sh run-group deepseek-int8
RUN_GROUP_WORKERS=1 bash Phase_2/VLMEvalKit/run_phase2.sh run-group vintern
RUN_GROUP_WORKERS=1 bash Phase_2/VLMEvalKit/run_phase2.sh run-group huatuo
```

Không submit bốn lệnh trên đồng thời vào cùng allocation. Chỉ submit group kế
tiếp sau khi group trước đã kết thúc hoặc đã được scheduler báo thành công.

### Bước 6 — Nếu DeepSeek vẫn báo thiếu memory

Đây là lỗi tài nguyên GPU, không phải lỗi package. Log cũ cho thấy GPU 0 còn
`69.64/94.97 GiB` nhưng vLLM cần khoảng `75.98 GiB`; cần allocation sạch hoặc
process đang chiếm GPU phải được admin xử lý. Không tự kill process người khác.

## Kết luận từ log `dermnet_vllm.o85842`

Có hai lỗi độc lập, không được gộp thành một lỗi memory:

1. Qwen3.5/Qwen3-VL chết khi khởi tạo worker vLLM, trước khi inference. Log chỉ
   vào `flashinfer/comm/fd_exchange.py` với annotation `array.array[int]` và
   Python 3.10 báo `TypeError: 'type' object is not subscriptable`. Đây là lỗi
   tương thích Python 3.10 với FlashInfer 0.6.16.
2. DeepSeek Small/Tiny có thêm lỗi bộ nhớ GPU: GPU 0 chỉ còn `69.64/94.97 GiB`
   nhưng vLLM yêu cầu khoảng `75.98 GiB` theo mức sử dụng 0.80. Log ghi nhận một
   process khác dùng khoảng 25 GiB. Script không tự dừng process của người khác.

Lỗi `LD_LIBRARY_PATH`/NVML là workaround cấp cluster đã biết: mọi process vLLM
được chạy qua launcher tạo bởi script với đúng dạng:

```bash
env -u LD_LIBRARY_PATH <python-trong-env> ...
```

## Ma trận thay đổi

| Luồng model | Environment | Script làm gì | Có đổi stack không? |
|---|---|---|---|
| Qwen3.5, Qwen3-VL | `dermnet-vllm-py312` mặc định | Tạo/reuse Python 3.12; pin `vllm==0.28.0`, `flashinfer-python==0.6.16.post3`, `flashinfer-cubin==0.6.16.post3`; tạo `dermnet-vllm-clean`; mapping chạy qua `env -u LD_LIBRARY_PATH` | Có |
| DeepSeek Small/Tiny vLLM | Dùng chung `dermnet-vllm-py312` | Hưởng cùng FlashInfer và launcher sạch; vẫn cần GPU trống khi khởi tạo | Không tạo env thứ hai |
| DeepSeek-VL2 8-bit | `dermnet-deepseek-int8` | Giữ Python/Torch/Transformers/bitsandbytes legacy; chạy `pip check`; runner sẽ kiểm tra CUDA và patch trong compute job | Không |
| Vintern 1B/3B | `dermnet-vintern` | Giữ stack legacy; chạy `pip check`; runner kiểm tra trong compute job | Không |
| HuatuoGPT-Vision | `dermnet-huatuo` | Giữ stack legacy; không ép Python 3.12/FlashInfer; runner kiểm tra source và patch trong compute job | Không |

Không áp dụng `env -u LD_LIBRARY_PATH` bừa cho ba environment legacy: log hiện tại
chỉ chứng minh workaround cần cho luồng vLLM/NVML. Nếu một legacy env sau này
phát sinh cùng lỗi, runner sẽ dừng ở preflight để kiểm tra riêng thay vì âm thầm
đổi thư viện của nó.

## Version phải cài

Mỗi profile bên dưới đã nạp thêm `Phase_2/VLMEvalKit/requirements.txt`. Các dòng
ghi `chưa pin` để người dùng biết repository hiện không yêu cầu một version cố
định cho package đó.

| Environment | Python | Package bắt buộc |
|---|---:|---|
| `dermnet-vllm-py312` | 3.12 | `vllm==0.28.0`; `flashinfer-python==0.6.16.post3`; `flashinfer-cubin==0.6.16.post3`; `torch==2.13.0`; `torchvision==0.28.0`; `transformers==5.17.0`; `qwen-vl-utils==0.0.14` |
| `dermnet-deepseek-int8` | 3.10 | `torch==2.8.0`; `torchvision==0.23.0`; `transformers==4.38.2`; `bitsandbytes==0.49.0`; `attrdict`, `einops`, `timm`, `sentencepiece` — chưa pin |
| `dermnet-vintern` | 3.10 | `torch==2.8.0`; `torchvision==0.23.0`; `transformers==4.42.3`; `einops`, `timm`, `sentencepiece` — chưa pin |
| `dermnet-huatuo` | 3.10 | `torch==2.8.0`; `torchvision==0.23.0`; `transformers==4.37.2`; `tokenizers==0.15.2`; `numpy==1.26.4`; `accelerate==0.21.0`; `bitsandbytes==0.49.0`; `einops==0.6.1`; `einops-exts==0.0.4`; `peft==0.4.0`; `sentencepiece==0.1.99`; `shortuuid==1.0.11`; `timm==0.6.13`; `markdown2==2.4.10`; `wavedrom==2.0.3.post3` |

Torch cho ba environment legacy phải lấy từ CUDA 12.8 index:

```bash
--index-url https://download.pytorch.org/whl/cu128
```

DeepSeek INT8 còn cần source `DeepSeek-VL2` đã được đưa vào `PYTHONPATH`; Huatuo
còn cần source `HuatuoGPT-Vision` và patch Blackwell. `prepare-runtime` xử lý hai
phần source này trên login node, không cài package trên run node.

## Chạy đúng trên cluster tách node

### 1. Login/setup node — có cài hoặc nâng package

Chạy một lần sau khi checkout code. Script có guard và sẽ từ chối nếu phát hiện
đang ở scheduler job thông dụng (`SLURM_JOB_ID`, `PBS_JOBID` hoặc `LSB_JOBID`).

```bash
cd /shared/homes/u26466553/projects/DermNet_Dataset
bash Phase_2/VLMEvalKit/scripts/fix_qwen_vllm_env.sh
```

Script mặc định sẽ:

- tạo `dermnet-vllm-py312` nếu chưa có;
- cài đúng requirement vLLM và cặp FlashInfer;
- tạo launcher `.../bin/dermnet-vllm-clean` có `env -u LD_LIBRARY_PATH`;
- cập nhật `Phase_2/VLMEvalKit/.phase2-server-env.sh` cho Qwen và DeepSeek
  Small/Tiny;
- chạy `pip check` cho cả ba environment legacy nhưng không cài/nâng chúng.

Nếu server dùng tên khác:

```bash
QWEN_VLLM_ENV=my-vllm-py312 \
DEEPSEEK_ENV=my-deepseek-int8 \
VINTERN_ENV=my-vintern \
HUATUO_ENV=my-huatuo \
  bash Phase_2/VLMEvalKit/scripts/fix_qwen_vllm_env.sh
```

Nếu chỉ muốn sửa Qwen và chưa có đủ ba env legacy, dùng `CHECK_ALL_ENVS=0`;
đây là lựa chọn rút gọn, không phải chế độ kiểm tra đầy đủ:

```bash
CHECK_ALL_ENVS=0 bash Phase_2/VLMEvalKit/scripts/fix_qwen_vllm_env.sh
```

Sau khi script kết thúc bằng `Environment fix passed`, package đã được chuẩn bị
trên login node. Không chạy script này trong job compute.

### 2. Run/compute node — chỉ kiểm tra và chạy model

Job không cài package. Sau khi login node đã tạo mapping, submit job chạy:

```bash
cd /shared/homes/u26466553/projects/DermNet_Dataset
bash Phase_2/VLMEvalKit/run_phase2.sh doctor
bash Phase_2/VLMEvalKit/run_phase2.sh run-group vllm
```

`doctor`/`run-group` dùng Python path tuyệt đối trong mapping, tự kiểm tra CUDA,
SM 12.0, package và memory trên chính compute node. Nếu GPU 0 còn process khác,
job phải chờ allocation sạch hoặc báo admin; không kill process tự động.

## Nếu còn lỗi memory DeepSeek

Đây không phải lỗi FlashInfer. Cách ổn định nhất là xin allocation có đủ hai GPU
trống và chạy lại `run-group vllm`. Chỉ khi admin xác nhận có thể dùng GPU đang
bận mới cân nhắc giảm `DERMNET_VLLM_GPU_UTIL`; biến này được dùng chung cho các
backend vLLM nên giảm nó có thể làm thay đổi cả Qwen và DeepSeek.

## Giới hạn của bản fix

Script xác nhận chắc chắn được lỗi import đã gặp: Python 3.12, đúng hai package
FlashInfer, `array[int]`, communicator vLLM và launcher không còn
`LD_LIBRARY_PATH`. Nó không thể xác nhận model đã sinh câu trả lời thành công
cho đến khi có một smoke test trên compute node với GPU sạch và model cache truy
cập được.
