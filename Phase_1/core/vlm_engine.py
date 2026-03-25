import torch
import gc
import os
import json
import re
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    AutoModel, 
    AutoTokenizer,
    BitsAndBytesConfig
)

# Thư viện hỗ trợ Qwen2-VL xử lý ảnh
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    os.system('pip install qwen-vl-utils')
    from qwen_vl_utils import process_vision_info

class VLMEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_id = None
        
        # Đường dẫn lưu trữ cố định trên Drive
        self.drive_cache_path = "/content/drive/MyDrive/DermNet_Dataset/huggingface_cache"
        
        # Cấu hình 4-bit
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def flush_memory(self):
        """Giải phóng hoàn toàn VRAM"""
        if self.model is not None: del self.model
        if self.processor is not None: del self.processor
        if self.tokenizer is not None: del self.tokenizer
        self.model = self.processor = self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"--- Đã giải phóng VRAM ---")

    def load_model(self, model_id: str):
        """Load model và tự động lưu/đọc từ Drive"""
        self.flush_memory()
        self.model_id = model_id
        os.makedirs(self.drive_cache_path, exist_ok=True)

        print(f"--- Đang nạp model: {model_id} ---")
        print(f"--- Cache Directory: {self.drive_cache_path} ---")

        common_kwargs = {
            "quantization_config": self.bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "cache_dir": self.drive_cache_path, # Lưu model vào Drive
            "attn_implementation": "sdpa"      # Fix lỗi Flash Attention
        }

        if "Qwen2-VL" in model_id:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, **common_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                model_id, cache_dir=self.drive_cache_path
            )
        
        elif "InternVL2" in model_id:
            self.model = AutoModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, **common_kwargs
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, use_fast=False, 
                cache_dir=self.drive_cache_path
            )

    def extract_json(self, raw_txt: str) -> dict:
        """Trích xuất JSON từ chuỗi văn bản."""
        match = re.search(r'```json\s*(.*?)\s*```', raw_txt, re.DOTALL)
        json_str = match.group(1) if match else raw_txt
        try:
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Không tìm thấy {}")
            return json.loads(json_str[start_idx:end_idx])
        except Exception as e:
            return {"error": "JSON parse failed", "details": str(e), "raw": raw_txt[:200]}

    def get_internvl_pixel_values(self, image_path):
        """Hàm helper chuẩn bị ảnh cho InternVL2"""
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

    def call_vlm(self, system_prompt: str, user_prompt: str, image_path: str = None) -> str:
        """Hàm inference đa model (Hỗ trợ System Prompt)"""
        if not self.model:
            return "ERROR: Model chưa được nạp!"

        if "Qwen2-VL" in self.model_id:
            # Qwen2-VL hỗ trợ role 'system' một cách native
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": []}
            ]
            
            if image_path:
                messages[1]["content"].append({"type": "image", "image": image_path})
            
            messages[1]["content"].append({"type": "text", "text": user_prompt})
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        elif "InternVL2" in self.model_id:
            # InternVL2 thường nhận prompt gộp cho system và user
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            if image_path:
                pixel_values = self.get_internvl_pixel_values(image_path)
                response, _ = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    full_prompt, 
                    generation_config={"max_new_tokens": 1024}
                )
            else:
                response, _ = self.model.chat(
                    self.tokenizer, 
                    None, 
                    full_prompt, 
                    generation_config={"max_new_tokens": 1024}
                )
            return response

# --- LOGIC PIPELINE (Cập nhật tham số system_prompt) ---

def run_phase1(engine: VLMEngine, image_path: str, sys_prompt: str, usr_prompt: str) -> str:
    print(f"[+] Phase 1: Observing {os.path.basename(image_path)}")
    return engine.call_vlm(sys_prompt, usr_prompt, image_path=image_path)

def run_phase2(engine: VLMEngine, phase1_output: str, sys_prompt: str, usr_template: str, disease_name: str, disease_knowledge: str) -> dict:
    print(f"[+] Phase 2: Mapping for {disease_name}")
    final_user_prompt = usr_template.format(
        phase1_qa_output=phase1_output,
        disease_name=disease_name,
        disease_knowledge=disease_knowledge
    )
    raw_response = engine.call_vlm(sys_prompt, final_user_prompt)
    return engine.extract_json(raw_response)

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Cấu trúc test mô phỏng y hệt file YAML của bạn
    SYS_P1 = "Bạn là chuyên gia. KHÔNG ĐOÁN TÊN BỆNH. Chỉ mô tả những gì thấy trên ảnh."
    USR_P1 = "Trả lời: 1. Loại tổn thương? 2. Màu sắc? 3. Hình dạng?"
    
    SYS_P2 = "Bạn là AI mapping. Luôn lấy tên bệnh làm Category. Trích xuất JSON mảng."
    USR_P2 = "DỮ LIỆU: \n1. Quan sát: {phase1_qa_output}\n2. Kiến thức: {disease_knowledge}\nXuất JSON."
    
    IMG_PATH = "Phase_1/assets/few_shot_image/Tinea_incognito.png" 
    
    engine = VLMEngine()

    # Chỉ chạy 1 model để test nhanh (Ví dụ: Qwen)
    try:
        if os.path.exists(IMG_PATH):
            engine.load_model("Qwen/Qwen2-VL-7B-Instruct")
            
            obs_qwen = run_phase1(engine, IMG_PATH, SYS_P1, USR_P1)
            print(f"\n[QWEN2-VL P1]:\n{obs_qwen}")
            
            json_qwen = run_phase2(engine, obs_qwen, SYS_P2, USR_P2, "Nấm da ẩn danh", "Có mảng hồng ban, mụn mủ")
            print(f"\n[QWEN2-VL P2 JSON]:\n{json.dumps(json_qwen, indent=2, ensure_ascii=False)}")
        else:
            print("[-] Không tìm thấy ảnh test, hãy sửa IMG_PATH!")
    except Exception as e:
        print(f"[-] Lỗi hệ thống: {e}")
    finally:
        engine.flush_memory()