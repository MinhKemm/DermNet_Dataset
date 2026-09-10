# Cài môi trường server từ đầu

Mục tiêu là sau khi clone chỉ thao tác qua một entrypoint:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh setup
bash Phase_2/VLMEvalKit/run_phase2.sh doctor
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Server mục tiêu là Linux, 2 NVIDIA RTX PRO 6000 Blackwell 96 GB, driver nhìn thấy CUDA 13.0. Dòng CUDA của `nvidia-smi` là khả năng của driver; phiên bản CUDA runtime thực tế đến từ wheel PyTorch/vLLM.

## 1. Phần có sẵn trên server

- Git và Conda trong `PATH`.
- NVIDIA driver hoạt động; `nvidia-smi` thấy GPU.
- CUDA 12.8 toolkit/compiler (`nvcc`) để build FlashAttention cho Huatuo với stack mặc định.
- Dung lượng đĩa đủ cho bốn environment, model cache và output.

Kiểm tra nhanh:

```bash
git --version
conda --version
nvidia-smi
nvcc --version
```

`nvidia-smi` có thể hiện CUDA 13.0 trong khi `nvcc` là 12.8; điều này bình thường. Driver 580 chạy được CUDA runtime 12.8, còn compiler phải khớp `torch.version.cuda`. Setup sẽ so sánh hai giá trị và dừng với hướng dẫn rõ nếu không khớp.

## 2. Setup tự động cài gì

`bash Phase_2/VLMEvalKit/run_phase2.sh setup` tạo bốn environment Python 3.10:

| Environment | Model | Backend chính |
|---|---|---|
| `dermnet-vllm` | Qwen3.5, Qwen3-VL, DeepSeek Small/Tiny | `vllm==0.28.0` |
| `dermnet-deepseek-int8` | DeepSeek-VL2 8-bit | Transformers 4.38.2, bitsandbytes 0.49.0 |
| `dermnet-vintern` | Vintern 1B/3B | Transformers 4.42.3 remote code |
| `dermnet-huatuo` | HuatuoGPT-Vision 34B | Transformers 4.37.2, FlashAttention 2.8.3.post1 |

Ba environment legacy cài mặc định PyTorch 2.8.0 + torchvision 0.23.0 từ CUDA 12.8 wheels. Driver CUDA 13 có khả năng tương thích ngược với runtime này. Có thể thay cặp wheel khi cụm HPC quy định stack khác:

```bash
LEGACY_TORCH='torch==PHIEN_BAN' \
LEGACY_TORCHVISION='torchvision==PHIEN_BAN' \
LEGACY_TORCH_INDEX_URL='URL_WHEEL_CUDA' \
bash Phase_2/VLMEvalKit/run_phase2.sh setup
```

Setup clone và ghim mã nguồn:

- DeepSeek-VL2 commit `ef9f91e2b6426536b83294c11742c27be66361b1`.
- HuatuoGPT-Vision commit `e1a52dcf6c0417f4b6ac1d378b01147280192fca`.

Sau cùng script sinh `Phase_2/VLMEvalKit/.phase2-server-env.sh`. Runner tự source file này, vì vậy `all` và `resume` dùng đúng Python mà không cần activate từng environment.

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
- DeepSeek 8-bit thiếu source `deepseek_vl2`, bitsandbytes hay xformers.
- Vintern thiếu torchvision/timm/sentencepiece hoặc Transformers quá cũ.
- Huatuo thiếu source chính thức, FlashAttention, peft hay dependency CLI.
- TSV, ảnh hoặc Excel nguồn dùng để vá bị thiếu.

Doctor mạnh hơn dry-run: dry-run chỉ in lệnh, còn doctor thực sự import từng backend và khởi tạo CUDA. Doctor chưa tải toàn bộ trọng số; phép thử cuối cùng vẫn là chạy inference thật.

## 5. Chạy và chạy tiếp

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
bash Phase_2/VLMEvalKit/run_phase2.sh all
```

Nếu SSH ngắt hoặc job bị dừng:

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Giữ nguyên checkout, thư mục output và `.phase2-server-env.sh`. Runner bỏ qua lượt đã có kết quả hoàn chỉnh và chạy lại lượt thiếu/thất bại.

## 6. Nguồn kỹ thuật

- [Cài vLLM trên NVIDIA GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Các model được vLLM hỗ trợ](https://docs.vllm.ai/en/stable/models/supported_models/)
- [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [DeepSeek-VL2 source](https://github.com/deepseek-ai/DeepSeek-VL2)
- [HuatuoGPT-Vision source](https://github.com/FreedomIntelligence/HuatuoGPT-Vision)
