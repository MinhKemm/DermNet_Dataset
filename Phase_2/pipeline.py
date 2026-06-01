import json
import requests
from itertools import cycle

class QwenLLM:
    """Module giao tiếp với Local Qwen qua Ollama"""
    def __init__(self, model_name="qwen3:4b", host="http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{host}/api/generate"

    def generate(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            print(f"Lỗi kết nối Qwen: {e}")
            return None

class PromptEngine:
    """Module quản lý Logic tạo Prompt theo các cấp độ nhận thức và loại câu hỏi"""
    def __init__(self):
        # Iterator xoay vòng vô tận qua 4 loại câu hỏi
        self.qtype_cycler = cycle(["Short Answer", "Multiple Choice", "Judgement", "Fill in the blank"])

    def _get_system_instruction(self, q_type):
        instructions = {
            "Short Answer": "Tạo 1 câu hỏi và trả lời ngắn gọn. Câu trả lời CHỈ chứa thông tin từ JSON.",
            "Multiple Choice": "Tạo 1 câu hỏi trắc nghiệm với 4 đáp án A, B, C, D. Phải tự tạo 3 đáp án sai hợp lý. Định dạng: Câu hỏi: ... Tùy chọn: A... B... C... D... Câu trả lời: [Ký tự đáp án đúng]",
            "Judgement": "Tạo 1 câu hỏi dạng Đúng/Sai (Có/Không). Có thể hỏi đúng sự thật hoặc cố tình hỏi sai để tạo đáp án 'Không'. Định dạng: Câu hỏi: ... Câu trả lời: Có/Không",
            "Fill in the blank": "Tạo 1 câu hỏi điền vào chỗ trống. Định dạng: Câu hỏi: ... [TRỐNG] ... Câu trả lời: ..."
        }
        return instructions.get(q_type, "")

    def build_anatomical_prompt(self, data, q_type):
        """Recognition: Phân bố (Bao gồm Vị trí và Kiểu phân bố)"""
        dist = ", ".join(data.get("Phan_bo", []))
        sys_inst = self._get_system_instruction(q_type)
        return f"""Bạn là một chuyên gia da liễu.
Nhiệm vụ: {sys_inst}
Dữ liệu gốc (đã được xác thực): Vị trí và phân bố của tổn thương là "{dist}".
Yêu cầu: Hãy đặt câu hỏi tập trung vào việc xác định vị trí trên cơ thể VÀ kiểu phân bố của tổn thương này."""

    def build_lesion_reasoning_prompt(self, data, q_type):
        """Understanding: Suy luận loại tổn thương từ các đặc điểm"""
        color = ", ".join(data.get("Mau_sac", []))
        shape = ", ".join(data.get("Hinh_dang", []))
        chars = ", ".join(data.get("Dac_diem", []))
        dist = ", ".join(data.get("Phan_bo", []))
        target_lesion = ", ".join(data.get("Loai_ton_thuong", []))
        
        sys_inst = self._get_system_instruction(q_type)
        return f"""Bạn là một chuyên gia da liễu.
Nhiệm vụ: {sys_inst}
Thông tin lâm sàng quan sát được: Màu sắc: {color}; Hình dạng: {shape}; Đặc điểm: {chars}; Phân bố: {dist}.
Loại tổn thương thực tế (Target): "{target_lesion}".
Yêu cầu: Hãy đặt câu hỏi yêu cầu chẩn đoán/xác định 'Loại tổn thương' dựa trên các thông tin lâm sàng đã cho."""

    def get_next_qtype(self):
        return next(self.qtype_cycler)

class VQADatasetBuilder:
    """Module Orchestrator kết nối luồng dữ liệu"""
    def __init__(self, llm_engine, prompt_engine):
        self.llm = llm_engine
        self.prompter = prompt_engine

    def process_single_json(self, json_data, image_id):
        data = json_data.get("TRICH_XUAT_JSON", {})
        if not data:
            return None

        # Lấy loại câu hỏi cho lượt này (đảm bảo 1 ảnh/1 JSON dùng 1 kiểu câu hỏi đồng nhất hoặc đổi kiểu tùy bạn)
        current_qtype = self.prompter.get_next_qtype()
        
        print(f"[{image_id}] Đang sinh dữ liệu theo format: {current_qtype}...")

        # 1. Task: Anatomical Distribution Recognition
        prompt_anatomy = self.prompter.build_anatomical_prompt(data, current_qtype)
        qa_anatomy = self.llm.generate(prompt_anatomy)

        # 2. Task: Lesion Reasoning
        prompt_reasoning = self.prompter.build_lesion_reasoning_prompt(data, current_qtype)
        qa_reasoning = self.llm.generate(prompt_reasoning)

        return {
            "image_id": image_id,
            "question_type": current_qtype,
            "tasks": {
                "Anatomical_Distribution": qa_anatomy,
                "Lesion_Reasoning": qa_reasoning
            }
        }

# ==========================================
# CÁCH CHẠY THỰC TẾ
# ==========================================
if __name__ == "__main__":
    # Khởi tạo pipeline
    # Hãy đảm bảo bạn đã bật terminal và chạy lệnh: `ollama run qwen3:4b`
    llm = QwenLLM(model_name="qwen3:4b") 
    prompter = PromptEngine()
    builder = VQADatasetBuilder(llm_engine=llm, prompt_engine=prompter)

    # Mock Data (Thay thế bằng file đọc từ thư mục DermNet_Data_V2 của bạn)
    sample_dataset = [
        {"image_id": "img_001.jpg", "TRICH_XUAT_JSON": {"Danh_muc_benh":"Acral lentiginous melanoma","Loai_ton_thuong":["Dát sắc tố"],"Mau_sac":["Nâu","Đen"],"Hinh_dang":["Không đều"],"Phan_bo":["Lòng bàn chân","Khu trú"],"Dac_diem":["Nhiều thùy","Ranh giới rõ"]}},
        {"image_id": "img_002.jpg", "TRICH_XUAT_JSON": {"Danh_muc_benh":"Eczema","Loai_ton_thuong":["Sẩn"],"Mau_sac":["Đỏ"],"Hinh_dang":["Tròn"],"Phan_bo":["Mặt gấp","Rải rác"],"Dac_diem":["Ngứa","Có vảy"]}}
    ]

    results = []
    for item in sample_dataset:
        qa_pair = builder.process_single_json(item, item["image_id"])
        if qa_pair:
            results.append(qa_pair)
            print(json.dumps(qa_pair, indent=2, ensure_ascii=False))
            print("-" * 50)
    
    # Lưu ra file JSONL chuẩn bị train VLM
    with open("DermNet_VQA_Generated.jsonl", "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")