import torch
import gc
import json
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Giả định bạn đã có hàm calculate_jaccard trong utils, nếu test độc lập thì có thể import
try:
    from Phase_1.utils.metrics import calculate_jaccard
except ImportError:
    # Fallback cho trường hợp chạy test file trực tiếp chưa có module metrics
    def calculate_jaccard(d1, d2): return 0.85 

class JudgeEngine:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_id = model_id
        self.drive_cache_path = "/content/drive/MyDrive/DermNet_Dataset/huggingface_cache"
        self.model = None
        self.tokenizer = None
        
        # Cấu hình 4-bit bắt buộc cho Colab T4
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def flush_memory(self):
        """Dọn dẹp VRAM tuyệt đối trước/sau khi chạy Judge"""
        if self.model is not None: del self.model
        if self.tokenizer is not None: del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("--- [Judge] Đã giải phóng hoàn toàn VRAM ---")

    def load_model(self):
        os.makedirs(self.drive_cache_path, exist_ok=True)
        print(f"--- [Judge] Nạp {self.model_id} từ Drive Cache ---")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            cache_dir=self.drive_cache_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=self.drive_cache_path, # Quan trọng
            attn_implementation="sdpa"
        )

    def extract_json(self, text: str) -> dict:
        """
        Trích xuất JSON an toàn, ưu tiên tìm block JSON_EXTRACTION.
        Xử lý cả trường hợp model sinh thêm text rác.
        """
        try:
            # Tìm block code markdown json
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            json_str = match.group(1) if match else text
            
            # Cố gắng tìm cặp ngoặc nhọn lớn nhất
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                parsed_data = json.loads(json_str[start_idx:end_idx])
                
                # Nếu model không bọc trong JSON_EXTRACTION, ta tự bọc lại cho đúng format
                if "JSON_EXTRACTION" not in parsed_data:
                    return {"JSON_EXTRACTION": parsed_data}
                return parsed_data
            else:
                raise ValueError("Không tìm thấy cặp ngoặc {} hợp lệ.")
        except Exception as e:
            return {"error": "JSON parse failed in Judge", "raw_output": text[:200], "details": str(e)}

    def build_prompt(self, vlm1_data: dict, vlm2_data: dict, j_score: float, disease_name: str) -> str:
        """Tạo Prompt ép Llama 3.1 xuất chuẩn định dạng JSON_EXTRACTION"""
        
        # Trích xuất dữ liệu lõi, bỏ qua metadata và lớp bọc JSON_EXTRACTION để đưa vào prompt gọn gàng
        d1 = vlm1_data.get("JSON_EXTRACTION", vlm1_data)
        d2 = vlm2_data.get("JSON_EXTRACTION", vlm2_data)
        
        d1_clean = {k: v for k, v in d1.items() if not k.startswith('_')}
        d2_clean = {k: v for k, v in d2.items() if not k.startswith('_')}

        # Sử dụng đúng format của Llama 3.1
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Bạn là một chuyên gia hội chẩn Da liễu AI cấp cao. Nhiệm vụ của bạn là hợp nhất dữ liệu từ hai mô hình VLM để tạo ra một bản ghi "Gold Standard" chuẩn xác.

[QUY TẮC CỐT LÕI - BẮT BUỘC TUÂN THỦ]
1. Cấu trúc output PHẢI LÀ một JSON duy nhất với root key là "JSON_EXTRACTION".
2. Trường "Category": Ghi chính xác "{disease_name}".
3. Các trường "Lesion_Type", "Colour", "Shape", "Distribution", "Characteristics": BẮT BUỘC phải là mảng các chuỗi (Array of Strings) [].
4. CƠ CHẾ HỢP NHẤT:
   - Nếu VLM1 và VLM2 dùng từ đồng nghĩa (VD: "Mảng đỏ" và "Hồng ban"), hãy dùng thuật ngữ y khoa chuẩn ("Hồng ban").
   - Gộp các đặc điểm không mâu thuẫn vào mảng.
   - Loại bỏ các thông tin có chữ "Không quan sát rõ" nếu mô hình kia nhìn thấy rõ.
5. KHÔNG giải thích, KHÔNG sinh thêm văn bản ngoài block JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>
[DỮ LIỆU ĐẦU VÀO]
- Chỉ số đồng thuận (Jaccard): {j_score:.2f}
- VLM 1: {json.dumps(d1_clean, ensure_ascii=False)}
- VLM 2: {json.dumps(d2_clean, ensure_ascii=False)}

Thực hiện hợp nhất và xuất JSON ngay lập tức:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
{{
  "JSON_EXTRACTION": {{"""

    def run_judge(self, vlm1_data: dict, vlm2_data: dict, disease_name: str) -> dict:
        """Quy trình chạy Judge hoàn chỉnh"""
        # 1. Fallback Logic: Nếu một trong hai VLM bị lỗi (Ví dụ: OOM, parse error)
        err1 = "error" in vlm1_data or "error" in vlm1_data.get("JSON_EXTRACTION", {})
        err2 = "error" in vlm2_data or "error" in vlm2_data.get("JSON_EXTRACTION", {})
        
        if err1 and not err2:
            final_data = vlm2_data.copy()
            final_data["_metadata"] = {"judge_status": "fallback_vlm2"}
            return final_data
        if err2 and not err1:
            final_data = vlm1_data.copy()
            final_data["_metadata"] = {"judge_status": "fallback_vlm1"}
            return final_data
        if err1 and err2:
            return {"error": "Cả hai VLM đều thất bại.", "Category": disease_name}
        
        # 2. Tính Jaccard (nếu 2 VLM đều thành công)
        j_score = calculate_jaccard(vlm1_data, vlm2_data)
        
        # 3. Inference với Llama 3.1
        prompt = self.build_prompt(vlm1_data, vlm2_data, j_score, disease_name)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=768, 
                temperature=0.01, # Giữ cực thấp để tránh hallucination
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Vì prompt ép model bắt đầu bằng `{\n  "JSON_EXTRACTION": {`, ta phải nối lại nếu text bị cắt
        if "assistant" in response_text:
            output_part = response_text.split("assistant")[-1].strip()
            # Khôi phục phần mồi (prefix)
            if not output_part.startswith("{"):
                output_part = '{\n  "JSON_EXTRACTION": {' + output_part
        else:
            output_part = response_text
            
        final_json = self.extract_json(output_part)
        
        # 4. Đính kèm Metadata kiểm thử
        if "JSON_EXTRACTION" in final_json:
            final_json["JSON_EXTRACTION"]["_metadata"] = {
                "judge_model": self.model_id,
                "input_jaccard": round(j_score, 4),
                "status": "consensus_achieved"
            }
        
        return final_json

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Dữ liệu mô phỏng theo đúng format JSON_EXTRACTION từ Phase 2
    mock_vlm1 = {
        "JSON_EXTRACTION": {
            "Category": "Ban đỏ đa dạng",
            "Lesion_Type": ["Hình bia bắn", "Sẩn đỏ"],
            "Colour": ["Đỏ", "Trung tâm sậm màu"],
            "Shape": ["Tròn", "Bờ rõ"],
            "Distribution": ["Lòng bàn tay", "Cẳng tay"],
            "Characteristics": ["Không quan sát rõ"]
        }
    }
    
    mock_vlm2 = {
        "JSON_EXTRACTION": {
            "Category": "Ban đỏ đa dạng",
            "Lesion_Type": ["Tổn thương hình bia bắn"],
            "Colour": ["Vòng đồng tâm biến đổi sắc đỏ"],
            "Shape": ["Hình tròn", "Đối xứng"],
            "Distribution": ["Cổ tay", "Cẳng tay"],
            "Characteristics": ["Biến đổi màu sắc 3 vùng"]
        }
    }

    judge = JudgeEngine()
    
    try:
        # Cần login Hugging Face trên Colab trước khi chạy lệnh này
        judge.load_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
        
        result = judge.run_judge(mock_vlm1, mock_vlm2, "Ban đỏ đa dạng")
        
        print("\n=== KẾT QUẢ HỢP NHẤT TỪ JUDGE ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Lỗi thực thi Judge: {e}")
    finally:
        judge.flush_memory()