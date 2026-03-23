# src/models/gpt4o_client.py
import base64
import os
from openai import OpenAI
from .base_client import BaseVLMClient

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

class GPT4oClient(BaseVLMClient):
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, image_path: str, system_prompt: str, few_shot_msgs: list, user_prompt: str) -> str:
        base64_image = encode_image(image_path)
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in few_shot_msgs:
            if msg['role'] == 'user':
                img_b64 = encode_image(msg['image_path'])
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg['text']},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                })
            elif msg['role'] == 'assistant':
                messages.append({"role": "assistant", "content": msg['text']})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        })

        try:
            res = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.config.get('max_tokens', 1024)
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"[GPT-4o] Lỗi: {e}")
            return ""