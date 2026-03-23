# src/models/qwen_client.py
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base_client import BaseVLMClient

class QwenClient(BaseVLMClient):
    def __init__(self, config: dict):
        super().__init__(config)
        print(f"Loading {self.model_id} on {config.get('device', 'cuda')}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, image_path: str, system_prompt: str, few_shot_msgs: list, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in few_shot_msgs:
            if msg['role'] == 'user':
                messages.append({
                    "role": "user",
                    "content": [{"type": "image", "image": msg['image_path']}, {"type": "text", "text": msg['text']}]
                })
            elif msg['role'] == 'assistant':
                messages.append({"role": "assistant", "content": msg['text']})

        messages.append({
            "role": "user",
            "content": [{"type": "image", "image": image_path}, {"type": "text", "text": user_prompt}]
        })

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.config.get('max_new_tokens', 1024), temperature=self.temperature
            )
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]