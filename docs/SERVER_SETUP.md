# Cài môi trường server từ đầu

Đây là tài liệu vận hành cho lệnh duy nhất dùng trên một server mới clone code. Đứng tại root `DermNet_Dataset` và chạy:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh server
```

Luồng thực tế của lệnh:

```text
server
  ├─ setup: tạo/cập nhật 4 Conda environment và ghi đường dẫn Python
  ├─ doctor: import backend, kiểm tra CUDA, dữ liệu và file vá
  └─ all: chạy tuần tự 16 lượt model × dataset tiếng Việt
```

Nếu một bước lỗi, script dừng ngay tại bước đó và chưa chuyển sang bước sau. Lệnh `setup`, `doctor` và `all` có thể chạy riêng để chẩn đoán, nhưng trên server mới phải bắt đầu bằng `server`.

Server mục tiêu là Linux, 2 NVIDIA RTX PRO 6000 Blackwell 96 GB, driver nhìn thấy CUDA 13.0. Dòng CUDA của `nvidia-smi` là khả năng của driver; phiên bản CUDA runtime thực tế đến từ wheel PyTorch/vLLM.

## 1. Phần có sẵn trên server

- Git và Conda trong `PATH`.
- NVIDIA driver hoạt động; `nvidia-smi` thấy GPU.
- Dung lượng đĩa đủ cho bốn environment, model cache và output.

Kiểm tra nhanh:

```bash
git --version
conda --version
nvidia-smi
```

Nếu `conda` đã cài nhưng SSH báo `command not found`, nạp script khởi tạo của bản Conda trên server rồi kiểm tra lại. Ví dụ với Conda đặt tại `/opt/conda`:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda --version
```

Đường dẫn Conda thực tế do quản trị viên server cung cấp. Không cần activate environment thủ công sau khi setup vì runner lưu chính xác đường dẫn Python của từng environment.

`nvidia-smi` có thể hiện CUDA 13.0 trong khi environment PyTorch dùng CUDA runtime khác; điều này bình thường. Driver 580 tương thích ngược với runtime CUDA trong wheel. Setup không biên dịch CUDA extension nên không yêu cầu `nvcc`.

## 2. Setup tự động cài gì

`bash Phase_2/VLMEvalKit/run_phase2.sh setup` tạo bốn environment Python 3.10:

| Environment | Model | Backend chính |
|---|---|---|
| `dermnet-vllm` | Qwen3.5, Qwen3-VL, DeepSeek Small/Tiny | vLLM 0.28.0, PyTorch 2.13.0 CUDA 13, Transformers 5.17.0 |
| `dermnet-deepseek-int8` | DeepSeek-VL2 8-bit | Transformers 4.38.2, bitsandbytes 0.49.0 |
| `dermnet-vintern` | Vintern 1B/3B | Transformers 4.42.3 remote code |
| `dermnet-huatuo` | HuatuoGPT-Vision 34B | Transformers 4.37.2, PyTorch eager attention |

Ba environment legacy khóa PyTorch 2.8.0 + torchvision 0.23.0 từ CUDA 12.8 wheels. Việc ghi phiên bản ở cả setup và profile ngăn lần chạy `pip` sau tự nâng Torch ngoài ý muốn. Driver CUDA 13 có khả năng tương thích ngược với runtime này. Có thể đổi mirror bằng `LEGACY_TORCH_INDEX_URL` nếu cụm HPC yêu cầu, nhưng mirror đó phải chứa đúng hai wheel đã khóa.

Setup clone và ghim mã nguồn:

- DeepSeek-VL2 commit `ef9f91e2b6426536b83294c11742c27be66361b1`.
- HuatuoGPT-Vision commit `e1a52dcf6c0417f4b6ac1d378b01147280192fca`.

Sau khi kiểm tra đúng commit, setup áp dụng hai bản vá nhỏ, có kiểm tra nội dung trước khi sửa:

- DeepSeek vision encoder dùng `torch.nn.functional.scaled_dot_product_attention`, để PyTorch chọn kernel tương thích SM 12.0 thay cho xFormers/FlashAttention bị ép trong source cũ.
- Huatuo chuyển cấu hình attention từ FlashAttention 2 bị ép sang `eager`, tương thích với Transformers 4.37.2 và không cần build CUDA extension.

Nếu source không đúng block đã duyệt, setup dừng thay vì sửa mù. Chạy setup lần nữa là an toàn vì bản vá có tính idempotent (đã vá thì giữ nguyên).

Sau cùng script sinh `Phase_2/VLMEvalKit/.phase2-server-env.sh`. Runner tự source file này, vì vậy `all` và `resume` dùng đúng Python mà không cần activate từng environment.

File mapping được tạo có vai trò như bảng định tuyến:

| Biến | Environment được trỏ tới | Model sử dụng |
|---|---|---|
| `PYTHON_QWEN` | `dermnet-vllm` | Hai Qwen |
| `PYTHON_DEEPSEEK_VLLM` | `dermnet-vllm` | DeepSeek Small/Tiny |
| `PYTHON_DEEPSEEK` | `dermnet-deepseek-int8` | DeepSeek 8-bit |
| `PYTHON_VINTERN` | `dermnet-vintern` | Vintern 1B/3B |
| `PYTHON_HUATUO` | `dermnet-huatuo` | Huatuo 34B |

Vì vậy Qwen không chạy bằng Python mặc định của shell. Nó chạy bằng Python trong `dermnet-vllm`, nơi setup đã cài `vllm`, PyTorch, Transformers và `qwen-vl-utils`.

## 3. Vì sao không dùng một requirements chung

Hai Qwen trong manifest khởi tạo nhánh vLLM thật; thiếu package `vllm` thì chắc chắn không chạy. DeepSeek Small/Tiny cũng gọi vLLM. Ngược lại DeepSeek 8-bit phải giữ Transformers + bitsandbytes, còn Vintern/Huatuo phụ thuộc các thế hệ Transformers cũ khác nhau. Ép bốn stack vào một environment sẽ tạo xung đột phiên bản.

`Phase_2/VLMEvalKit/requirements.txt` chỉ chứa dependency lõi. Các dependency chạy model được khóa trong:

```text
Phase_2/VLMEvalKit/requirements/server/vllm-blackwell.txt
Phase_2/VLMEvalKit/requirements/server/deepseek-int8-blackwell.txt
Phase_2/VLMEvalKit/requirements/server/vintern-blackwell.txt
Phase_2/VLMEvalKit/requirements/server/huatuo-blackwell.txt
```

Không dùng riêng `pip install -r requirements.txt` để kết luận server đã sẵn sàng.

## 4. Doctor kiểm tra gì

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh doctor
```

Doctor dừng ngay nếu gặp một trong các lỗi:

- Python mapping bị thiếu hoặc sai environment.
- PyTorch không thấy CUDA hoặc wheel không có kernel SM 12.0 cho Blackwell.
- Qwen/DeepSeek vLLM thiếu `vllm>=0.28,<0.29` hoặc Transformers phù hợp.
- DeepSeek 8-bit thiếu source `deepseek_vl2` hay bitsandbytes.
- Vintern thiếu torchvision/timm/sentencepiece hoặc Transformers quá cũ.
- Huatuo thiếu source chính thức, peft hay dependency CLI.
- TSV, ảnh hoặc Excel nguồn dùng để vá bị thiếu.

Doctor mạnh hơn dry-run: dry-run chỉ in lệnh, còn doctor thực sự import từng backend và khởi tạo CUDA. Doctor chưa tải toàn bộ trọng số; phép thử cuối cùng vẫn là chạy inference thật.

### Kiểm tra riêng Qwen và vLLM

Lỗi `ModuleNotFoundError: No module named 'vllm'` nghĩa là Qwen đang chạy bằng Python chưa được setup hoặc lệnh `setup` trước đó chưa hoàn tất. Kiểm tra trực tiếp environment mà runner tạo:

```bash
conda run -n dermnet-vllm python -c 'import vllm, torch, transformers, qwen_vl_utils; print("vLLM", vllm.__version__); print("Torch", torch.__version__); print("CUDA", torch.version.cuda, torch.cuda.is_available()); print("Transformers", transformers.__version__)'
```

Sau đó chạy doctor qua đúng entrypoint:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh doctor
```

Doctor chỉ báo đạt khi environment vLLM thỏa phiên bản đã khóa và PyTorch truy cập được GPU. Không cần tự chạy `pip install vllm` vào environment hiện tại của shell.

## 5. Chạy và chạy tiếp

### Máy mới clone hoặc setup chưa hoàn tất

```bash
git clone https://github.com/MinhKemm/DermNet_Dataset.git
cd DermNet_Dataset
bash Phase_2/VLMEvalKit/run_phase2.sh server
```

Nếu cần tải model gated/private, đặt `HF_TOKEN` trong secret của job hoặc export cho phiên SSH trước khi chạy `server`. Không ghi token vào repository.

Nếu mạng hoặc SSH ngắt khi đang cài package, chạy lại cùng lệnh `server`. Setup có thể chạy lại: Conda giữ environment đã tạo, `pip` hoàn thiện các package còn thiếu, source model được kiểm tra theo commit đã ghim, rồi doctor chạy lại trước inference.

### Environment đã setup và muốn kiểm tra trước khi chạy

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
bash Phase_2/VLMEvalKit/run_phase2.sh doctor
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

`plan` chỉ liệt kê 16 lượt. `doctor` kiểm tra môi trường và dữ liệu. `all` bắt đầu toàn bộ kế hoạch nhưng giữ lại những kết quả đã hoàn chỉnh theo cơ chế checkpoint.

### Inference đã chạy nhưng bị gián đoạn

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Giữ nguyên checkout, thư mục output và `.phase2-server-env.sh`. Runner bỏ qua lượt đã có kết quả hoàn chỉnh và chạy lại lượt thiếu/thất bại. Không xóa thư mục `outputs/answer-format-v4-vllm` trước khi resume.

## 6. Đọc log và xác định bước lỗi

- Lỗi trước dòng `Doctor passed`: vấn đề thuộc environment, CUDA, dữ liệu, ảnh hoặc Excel nguồn; sửa nguyên nhân rồi chạy lại `server`.
- Lỗi `No module named 'vllm'`: chạy lại `server` hoặc `setup`, sau đó dùng lệnh kiểm tra Qwen ở trên.
- Lỗi tải checkpoint từ Hugging Face: kiểm tra mạng, dung lượng cache và quyền của `HF_TOKEN`.
- CUDA out of memory: kiểm tra `nvidia-smi` xem GPU có tiến trình khác; runner không tự dừng tiến trình của người dùng khác.
- SSH ngắt trong lúc inference: kết nối lại và dùng `resume`.

Log từng lượt và trạng thái checkpoint nằm trong:

```text
Phase_2/VLMEvalKit/outputs/answer-format-v4-vllm/.phase2-runner/
```

## 7. Nguồn kỹ thuật

- [Cài vLLM trên NVIDIA GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Các model được vLLM hỗ trợ](https://docs.vllm.ai/en/stable/models/supported_models/)
- [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [DeepSeek-VL2 source](https://github.com/deepseek-ai/DeepSeek-VL2)
- [HuatuoGPT-Vision source](https://github.com/FreedomIntelligence/HuatuoGPT-Vision)
