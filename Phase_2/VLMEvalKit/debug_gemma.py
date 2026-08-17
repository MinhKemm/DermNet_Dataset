import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

image_path = '../../dermnet-output/images/Acne vulgaris/190.jpg'
model_path = 'google/gemma-4-31B-it'

print("="*50)
print(" STAGE 1: KIỂM TRA ẢNH RAW QUA PIL & NUMPY")
print("="*50)
try:
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    print(f"👉 Load ảnh thành công! Kích thước: {img.size}")
    print(f"👉 Giá trị Pixel - Nhỏ nhất (Min): {img_np.min()}, Lớn nhất (Max): {img_np.max()}")
    print(f"👉 Giá trị Pixel trung bình (Mean): {img_np.mean():.2f}")
    if img_np.min() == img_np.max():
        print("❌ CẢNH BÁO: Ảnh thô thực chất là một màu đồng nhất! Hãy kiểm tra lại file ảnh gốc trên đĩa.")
    else:
        print("✅ Ảnh thô có chi tiết màu sắc bình thường.")
except Exception as e:
    print(f"❌ Lỗi đọc ảnh: {e}")
    exit()

print("\n" + "="*50)
print(" STAGE 2: TẢI MODEL VÀ PROCESSOR (8-BIT)")
print("="*50)
# Cấu hình NF4 tối ưu bộ nhớ và giữ nguyên ma trận thị giác
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    quantization_config=quantization_config, 
    device_map="auto", 
    trust_remote_code=True
).eval()
print("✅ Tải Model và Processor thành công.")

print("\n" + "="*50)
print(" STAGE 3: KIỂM TRA CHAT TEMPLATE & CẤU TRÚC PROMPT")
print("="*50)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Dựa vào hình ảnh trên, hình dạng hoặc ranh giới của tổn thương được mô tả như thế nào?"}
        ]
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("--- CHUỖI PROMPT TRUYỀN VÀO MODEL ---")
print(prompt)
print("-------------------------------------")

print("\n" + "="*50)
print(" STAGE 4: ÉP TRÍCH XUẤT MẠNG SIGLIP (GEMMA 4)")
print("="*50)

# 1. Khởi tạo inputs thông qua template chính chủ
inputs = processor.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=True,             
    return_dict=True, 
    return_tensors="pt"
).to("cuda")

# 2. Ép kiểu dữ liệu
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

# 3. ÉP BỘ TRÍCH XUẤT CHUẨN HÓA ĐỘC LẬP
try:
    # Gọi trực tiếp bộ tiền xử lý hình ảnh ẩn (Image Processor) của Gemma 4
    image_processor = processor.image_processor
    
    # Tiền xử lý ảnh thô tách biệt khỏi luồng văn bản
    processed_image_dict = image_processor(img, return_tensors="pt")
    
    # Lấy ma trận pixel thực tế chưa bị biến đổi thành vector nhúng ẩn
    siglip_pixels = processed_image_dict["pixel_values"].squeeze(0).numpy()
    
    print(f"✅ Đã tóm được ma trận Siglip ban đầu! Shape: {siglip_pixels.shape}")
    
    # Đổi trục ma trận từ [Channels, Height, Width] thành [Height, Width, Channels]
    siglip_pixels = np.transpose(siglip_pixels, (1, 2, 0))
    
    # Khôi phục dải màu từ chuẩn hóa Siglip (thường dùng Mean/Std cụ thể)
    # Ta chuẩn hóa min-max nhanh để kiểm tra trực quan
    p_min, p_max = siglip_pixels.min(), siglip_pixels.max()
    print(f"👉 Giá trị ma trận ảnh Siglip - Min: {p_min:.4f}, Max: {p_max:.4f}")
    
    if p_min == p_max:
        print("❌ CẢNH BÁO: Ma trận Siglip bị triệt tiêu hoàn toàn thành mảng phẳng!")
    else:
        # Đưa về dải 0-255 thông thường
        normalized_pixels = (siglip_pixels - p_min) / (p_max - p_min)
        u8_pixels = (normalized_pixels * 255).astype(np.uint8)
        
        # Ghi file ảnh ra đĩa
        debug_path = "model_vision_input.png"
        Image.fromarray(u8_pixels).save(debug_path)
        print(f"📸 ĐÃ LƯU FILE THÀNH CÔNG! Hãy kiểm tra file '{debug_path}' trong thư mục hiện tại.")

except Exception as e:
    print(f"❌ Không thể can thiệp sâu vào Image Processor: {e}")


print("\n" + "="*50)
print(" STAGE 5: CHẠY INFERENCE VÀ KIỂM TRA KẾT QUẢ")
print("="*50)
input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids = generated_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

print("👉 KẾT QUẢ ĐẦU RA CHÍNH THỨC:")
print(response)
print("="*50)