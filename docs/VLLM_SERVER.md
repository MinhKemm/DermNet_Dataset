# Chạy một file trên Blackwell

Ảnh server do người dùng cung cấp: 2 NVIDIA RTX PRO 6000 Blackwell Server Edition, khoảng 96 GB/GPU, driver 580.142. Dòng CUDA 13.0 của nvidia-smi là khả năng driver, không xác nhận phiên bản torch/CUDA runtime đang cài. Dùng stack vLLM/PyTorch hỗ trợ Blackwell; không dùng torch 2.0.1 từ requirements legacy một cách máy móc. Xem [cài GPU chính thức](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).

## Backend thực tế

| Model | Backend hiện tại |
|---|---|
| Qwen3.5-35B-A3B, Qwen3-VL-8B | vLLM, đã có nhánh thật |
| DeepSeek Small, Tiny BF16 | vLLM, nhánh mới gọi LLM.generate với ảnh và prompt |
| DeepSeek 8-bit | Transformers/BitsAndBytes, giữ đúng lượng tử hóa yêu cầu |
| Hai Vintern | Transformers, adapter hiện chưa có nhánh vLLM |
| Huatuo 34B | CLI chính thức, adapter hiện chưa có nhánh vLLM |

Không đổi cờ cho mọi class một cách cơ học. Gemma trong ảnh đã bị loại khỏi danh sách chạy. Danh sách [vLLM hỗ trợ](https://docs.vllm.ai/en/stable/models/supported_models/) có DeepSeek-VL2 và InternVL, nhưng hỗ trợ kiến trúc không tự tạo nhánh inference cho adapter Vintern của repo này. Chuyển bản 8-bit sang backend khác cần xác nhận đúng định dạng lượng tử hóa; không tự thay bằng 4-bit/BF16.

DeepSeek mới dùng mẫu prompt từ [ví dụ chính thức vLLM](https://github.com/vllm-project/vllm/blob/v0.10.2/examples/offline_inference/vision_language.py), giới hạn một ảnh/câu, context 4096 và sinh tối đa 512 token, BF16. Câu hỏi cùng chỉ dẫn định dạng của dataset được giữ nguyên. Không hỗ trợ hội thoại nhiều lượt trong nhánh này; đầu vào không phù hợp báo lỗi. Đây là thay đổi backend/generation của đợt vá; không khẳng định tái lập nguyên trạng kết quả lịch sử.

## Dùng chung một file điều phối

Sau khi chuẩn bị các môi trường và nạp các biến Python theo [setup](SERVER_SETUP.md):

```bash
# Small/Tiny dùng môi trường vLLM giống Qwen, bản int8 vẫn dùng PYTHON_DEEPSEEK:
export PYTHON_DEEPSEEK_VLLM="$PYTHON_QWEN"
export DERMNET_VLLM_GPU_UTIL=0.80
bash Phase_2/VLMEvalKit/run_phase2.sh all
# Sau gián đoạn, giữ cùng biến môi trường và thư mục kết quả:
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Mặc định 8 model/16 lượt (12 full + 4 vá). `PYTHON_DEEPSEEK_VLLM` mặc định bằng `PYTHON_QWEN`. DeepSeek Small/Tiny không cần package `deepseek_vl2` cũ khi chạy nhánh vLLM. Bản 8-bit vẫn cần package này và bitsandbytes.

Kết quả mặc định đổi sang `outputs/answer-format-v4-vllm` để không dùng checkpoint trước khi đổi backend. Không đặt RUN_WORK_DIR về v3 để tái dùng kết quả cũ. Thư mục Vintern chạy mới nằm bên trong root kết quả này.

## LLaVA đã bỏ

Luồng mặc định không còn LLaVA-med và không cần SKIP_LLAVA. Giữ nguyên 8 model còn lại. Huatuo vẫn dùng module llava của tác giả; không xóa module đó.

## Giới hạn kiểm chứng

40 kiểm thử đã đạt, có mô phỏng API vLLM và dry-run 16 lượt. Chưa nạp model thật trên GPU server. Các môi trường legacy vẫn cần thử trên Blackwell. Ảnh cho thấy GPU đang có tiến trình khác; 80% là tỷ lệ cấp phát tối đa theo tổng VRAM, không bảo đảm luôn đủ bộ nhớ trống. Chạy trên compute node được cấp GPU, giữ CUDA_VISIBLE_DEVICES do hệ thống cấp; không tự dừng tiến trình người khác.
