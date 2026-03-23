# src/core/prompt_builder.py
import os

class PromptBuilder:
    def __init__(self, config: dict, assets_dir: str):
        self.config = config
        self.assets_dir = assets_dir

    def get_system_prompt(self) -> str:
        return self.config['system_instruction'] + "\n\n" + self.config['cot_steps']

    def get_few_shot_messages(self) -> list:
        msgs = []
        if 'few_shot_examples' not in self.config:
            return msgs
            
        for ex_dict in self.config['few_shot_examples']:
            ex = list(ex_dict.values())[0]
            
            # 1. Lấy đường dẫn ảnh
            img_path = os.path.join(self.assets_dir, ex['image_name'])
            
            # 2. ĐỌC FILE KIẾN THỨC BỆNH (Phần mới thêm)
            knowledge_text = "Not available"
            knowledge_path = os.path.join(self.assets_dir, ex.get('disease_knowledge_path', ''))
            
            if os.path.exists(knowledge_path) and os.path.isfile(knowledge_path):
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    knowledge_text = f.read().strip()
            else:
                print(f"⚠️ Cảnh báo: Không tìm thấy file kiến thức few-shot tại {knowledge_path}")

            clinical_desc = ex.get('clinical_desc', 'Not available')
            
            # 3. Ghép vào prompt
            user_text = f"DISEASE_KNOWLEDGE:\n{knowledge_text}\n\nCASE_INPUT:\n{clinical_desc}"
            
            # 4. Đưa vào tin nhắn
            msgs.append({"role": "user", "image_path": img_path, "text": user_text})
            msgs.append({"role": "assistant", "text": ex['expected_json']})
            
        return msgs

    def get_user_prompt(self, knowledge: str, clinical: str) -> str:
        return self.config['user_template'].format(
            disease_knowledge=knowledge, 
            clinical_description=clinical
        )