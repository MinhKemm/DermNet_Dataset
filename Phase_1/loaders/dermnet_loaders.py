# src/loaders/dermnet_loader.py
import os
import glob

class DermNetLoader:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục gốc: {self.root_dir}")

    def get_all_samples(self) -> list:
        samples = []
        for disease_name in os.listdir(self.root_dir):
            disease_path = os.path.join(self.root_dir, disease_name)
            if not os.path.isdir(disease_path):
                continue
            
            # 1. Đọc file kiến thức bệnh ("Toàn bộ nội dung...")
            disease_knowledge = "Not available"
            knowledge_files = glob.glob(os.path.join(disease_path, "Toàn bộ nội dung*.txt"))
            if knowledge_files:
                with open(knowledge_files[0], 'r', encoding='utf-8') as f:
                    disease_knowledge = f.read().strip()
            
            # 2. Quét thư mục images/
            images_dir = os.path.join(disease_path, "images")
            if not os.path.exists(images_dir):
                continue
                
            for file_name in os.listdir(images_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(images_dir, file_name)
                    base_name = os.path.splitext(file_name)[0]
                    
                    # 3. Đọc mô tả lâm sàng riêng của ảnh
                    txt_path = os.path.join(images_dir, f"{base_name}.txt")
                    clinical_desc = "Not available"
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            clinical_desc = f.read().strip()
                            
                    samples.append({
                        "sample_id": f"{disease_name}_{base_name}",
                        "disease_name": disease_name,
                        "image_path": img_path,
                        "disease_knowledge": disease_knowledge,
                        "clinical_description": clinical_desc
                    })
        return samples