import os
from pathlib import Path
from typing import List

def get_project_root() -> Path:
    """Tự động phát hiện thư mục gốc của dự án"""
    return Path(__file__).parent.parent.parent

def load_disease_knowledge(disease_name_en: str, contents_dir: str) -> str:
    """Đọc file kiến thức (.txt) của bệnh"""
    file_path = Path(contents_dir) / f"Toàn bộ nội dung - {disease_name_en}.txt"
    if not file_path.exists():
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_image_paths(disease_name_en: str, images_dir: str) -> List[str]:
    """Trả về danh sách đường dẫn tất cả các ảnh trong thư mục bệnh"""
    disease_img_dir = Path(images_dir) / disease_name_en
    if not disease_img_dir.exists():
        return []
    
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = []
    for f in disease_img_dir.iterdir():
        if f.is_file() and f.suffix.lower() in valid_exts:
            images.append(str(f))
    return sorted(images)

def get_all_diseases(images_dir: str) -> List[str]:
    """Trả về danh sách tên các thư mục bệnh"""
    dir_path = Path(images_dir)
    if not dir_path.exists():
        return []
        
    diseases = []
    for d in dir_path.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            diseases.append(d.name)
    return sorted(diseases)