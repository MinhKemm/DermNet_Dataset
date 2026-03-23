# src/models/ollama_client.py
import ollama
from .base_client import BaseVLMClient

class OllamaClient(BaseVLMClient):
    def __init__(self, config: dict):
        super().__init__(config)
        print(f"🔌 Khởi tạo kết nối Local Ollama với model: {self.model_id}")
        
        # Pull model tự động nếu trên máy chưa có
        try:
            ollama.show(self.model_id)
        except ollama.ResponseError:
            print(f"⏳ Đang tải {self.model_id} về máy (chỉ một lần đầu)...")
            ollama.pull(self.model_id)

    def generate(self, image_path: str, system_prompt: str, few_shot_msgs: list, user_prompt: str) -> str:
        # Cấu trúc messages chuẩn của Ollama
        messages = [{"role": "system", "content": system_prompt}]
        
        # Nạp Few-shot
        for msg in few_shot_msgs:
            if msg['role'] == 'user':
                messages.append({
                    "role": "user",
                    "content": msg['text'],
                    "images": [msg['image_path']] # Ollama nhận luôn đường dẫn ảnh
                })
            elif msg['role'] == 'assistant':
                messages.append({"role": "assistant", "content": msg['text']})

        # Nạp User Input hiện tại
        # Nếu dùng cho Phase 2 (LLM Llama) thì image_path sẽ là None
        current_msg = {"role": "user", "content": user_prompt}
        if image_path:
            current_msg["images"] = [image_path]
            
        messages.append(current_msg)

        try:
            # Gọi API Local của Ollama
            response = ollama.chat(
                model=self.model_id,
                messages=messages,
                options={"temperature": self.temperature}
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ [Ollama Error - {self.model_id}]: {e}")
            return ""