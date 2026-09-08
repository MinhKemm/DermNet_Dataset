# Cài môi trường server

**Cập nhật:** [cấu hình Blackwell/vLLM](VLLM_SERVER.md) thay thế phần chọn backend bên dưới cho DeepSeek Small/Tiny: dùng `PYTHON_DEEPSEEK_VLLM`, mặc định bằng `PYTHON_QWEN`. Môi trường DeepSeek legacy bên dưới chỉ còn phục vụ bản 8-bit. Một file điều phối không yêu cầu mọi model dùng chung môi trường.

Hướng dẫn cho Linux/Bash, server NVIDIA 2 × 96 GB. Đây là quy trình chuẩn bị và kiểm tra; chưa phải bộ phiên bản đã chạy inference thành công trên server đích. VRAM không cho biết kiến trúc GPU hoặc phiên bản CUDA cần dùng.

## 1. Chuẩn bị checkout và kiểm tra GPU

```bash
git clone https://github.com/MinhKemm/DermNet_Dataset.git
cd DermNet_Dataset
export DERMNET_ROOT="$PWD"
export DERMNET_KIT="$DERMNET_ROOT/Phase_2/VLMEvalKit"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Nếu đã clone thì dùng checkout hiện có. Kiểm tra đủ hai TSV Việt, ảnh và bốn Excel DeepSeek trong `outputs/*/source/`. Phiên bản code và Excel mới phải được push trước khi server clone để có những thay đổi này.

## 2. Tạo môi trường riêng

Nếu quản trị viên đã có môi trường phù hợp, bỏ qua tạo mới và dùng đường dẫn Python tương ứng. Ví dụ với Conda đã cài:

```bash
conda create -n dermnet-qwen python=3.11 pip -y
conda create -n dermnet-deepseek python=3.10 pip -y
conda create -n dermnet-vintern python=3.10 pip -y
conda create -n dermnet-huatuo python=3.10 pip -y
```

| Environment | Runner dùng biến | Model |
|---|---|---|
| dermnet-qwen | PYTHON_QWEN | Hai Qwen3/3.5 |
| dermnet-deepseek | PYTHON_DEEPSEEK | Ba DeepSeek |
| dermnet-vintern | PYTHON_VINTERN | Hai Vintern |
| dermnet-huatuo | PYTHON_HUATUO | Huatuo 34B |

Qwen cần stack mới; các model khác dùng mã Transformers cũ. Huatuo và LLaVA-med đều cung cấp module tên `llava`, nên phải tách môi trường. Vintern có Python riêng để không bị khóa theo LLaVA-med.

## 3. PyTorch và dependencies

Với mỗi môi trường ngoài Qwen, kích hoạt rồi cài **cặp torch/torchvision có CUDA phù hợp GPU và driver**, dùng lệnh được chọn tại [PyTorch](https://pytorch.org/get-started/locally/). Không sao chép pin torch 2.0.1 từ repo cũ nếu GPU mới không được hỗ trợ. Qwen dùng cặp torch do vLLM yêu cầu. Không đổi driver hệ thống chỉ để thử một phiên bản package.

Sau khi cài torch, mỗi environment phải vượt qua:

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); x=torch.ones(4, device='cuda'); print(x.sum().item(), torch.cuda.get_device_name(0))"
```

Từ root repo, các lệnh dưới đây cài dependencies VLMEvalKit cùng yêu cầu model. Nếu resolver báo xung đột, dừng và giải quyết trong đúng environment; không dùng `--no-deps` để che lỗi của VLMEvalKit.

### Qwen

```bash
conda activate dermnet-qwen
python -m pip install --upgrade pip uv
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
python -m pip install -e "$DERMNET_KIT" qwen-vl-utils
python -m pip check
python -c "from transformers import AutoConfig; import vllm; print(AutoConfig.from_pretrained('Qwen/Qwen3.5-35B-A3B').model_type)"
export PYTHON_QWEN="$(command -v python)"
```

Lệnh vLLM theo [model card Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-35B-A3B). Nightly thay đổi theo thời gian; sau khi chạy thử đạt cần lưu `pip freeze`. Không tự nâng Transformers độc lập nếu phiên bản đó xung đột với vLLM.

### DeepSeek

```bash
conda activate dermnet-deepseek
python -m pip install -e "$DERMNET_KIT" "transformers==4.38.2" bitsandbytes attrdict einops timm sentencepiece
mkdir -p "$DERMNET_ROOT/vendor"
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git "$DERMNET_ROOT/vendor/DeepSeek-VL2"
python -m pip install --no-deps -e "$DERMNET_ROOT/vendor/DeepSeek-VL2"
python -c "from deepseek_vl2.models import DeepseekVLV2Processor; import bitsandbytes; print('DeepSeek imports OK')"
export PYTHON_DEEPSEEK="$(command -v python)"
```

Transformers 4.38.2 lấy từ [requirements chính thức](https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/requirements.txt). `--no-deps` ở bước đăng ký source DeepSeek nhằm tránh tự thay torch đã chuẩn bị bằng torch 2.0.1 của upstream; các dependencies khác thiếu vẫn phải xử lý. Nếu mã upstream yêu cầu xformers, chọn bản cùng torch/CUDA, không cài tùy ý bản mới nhất.

### Vintern

```bash
conda activate dermnet-vintern
python -m pip install -e "$DERMNET_KIT" "transformers==4.37.2" timm einops sentencepiece
python -m pip check
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('5CD-AI/Vintern-3B-beta', trust_remote_code=True).model_type)"
export PYTHON_VINTERN="$(command -v python)"
```

4.37.2 là điểm bắt đầu cho stack legacy, không phải lockfile đã kiểm chứng cho cả hai Vintern. Đối chiếu [model card tác giả](https://huggingface.co/5CD-AI/Vintern-3B-beta) và thử cả hai model. `trust_remote_code` thực thi mã từ repo model; chỉ dùng nguồn đã tin cậy. Chỉ cài flash-attn nếu đường nạp model thực tế yêu cầu và bản đó khớp torch/CUDA.

### Huatuo 34B

```bash
conda activate dermnet-huatuo
git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git "$DERMNET_ROOT/vendor/HuatuoGPT-Vision"
export HUATUO_SOURCE_DIR="$DERMNET_ROOT/vendor/HuatuoGPT-Vision"
python -m pip install -e "$DERMNET_KIT" "transformers==4.37.2" "tokenizers>=0.14,<0.19" "numpy<2" "timm==0.6.13" "peft==0.4.0" einops-exts shortuuid markdown2 wavedrom
export PYTHON_HUATUO="$(command -v python)"
PYTHONPATH="$HUATUO_SOURCE_DIR" python -c "from cli import HuatuoChatbot; print('Huatuo imports OK')"
python -m pip check
```

Không cài bản `llava` khác trong môi trường này. [Requirements Huatuo](https://github.com/FreedomIntelligence/HuatuoGPT-Vision/blob/main/requirements.txt) đồng thời khóa Transformers 4.37.2 và tokenizers 0.13.3; hướng dẫn trên để resolver chọn tokenizers phù hợp Transformers. Không cài các package huấn luyện như deepspeed chỉ để chạy CLI inference nếu không được yêu cầu bởi đường import.

## 4. Giữ cấu hình Python cho lần chạy sau

Các lệnh `export` ở trên có hiệu lực trong phiên shell hiện tại. Ghi lại đường dẫn tuyệt đối được in bởi:

```bash
printf '%s\n' "$PYTHON_QWEN" "$PYTHON_DEEPSEEK" "$PYTHON_LEGACY" "$PYTHON_VINTERN" "$PYTHON_HUATUO" "$HUATUO_SOURCE_DIR"
```

Đưa các dòng `export TEN_BIEN=/duong/dan/thuc` vào file riêng ngoài Git, ví dụ `/srv/dermnet-env.sh`. Trước `all` hoặc `resume`, chạy `source /srv/dermnet-env.sh`. Không chỉ lưu tên environment: runner cần đường dẫn executable. Nếu cần token tải model, đặt `HF_TOKEN` qua cơ chế secret của server, không ghi vào README/Git.

## 5. Kiểm tra rồi chạy

Trong **từng** environment, từ thư mục VLMEvalKit:

```bash
cd "$DERMNET_KIT"
python -m pip check
python -c "import vlmeval; import pandas, openpyxl; print('VLMEvalKit imports OK')"
python run.py --help
```

Nếu `pip check` của source legacy báo pin torch/demo khác stack hiện có, ghi lại và đánh giá từng mục; không coi import thành công là bằng chứng mọi dependency tương thích. Lưu `python -m pip freeze` và commit của các repo vendor sau khi đã chạy model thật đạt.

```bash
cd "$DERMNET_ROOT"
bash Phase_2/VLMEvalKit/run_phase2.sh plan
DRY_RUN=1 bash Phase_2/VLMEvalKit/run_phase2.sh all
bash Phase_2/VLMEvalKit/run_phase2.sh all
# Sau gián đoạn, nạp lại cùng các biến Python và chạy:
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Dry-run chỉ kiểm tra lệnh/đường dẫn, không nạp model. `all` kiểm tra import trước inference; để xác nhận môi trường ổn cần thử tải trọng số và sinh câu trả lời của cả 8 model trên GPU. Không có cam kết chạy trọn bộ chỉ từ kết quả dry-run.
