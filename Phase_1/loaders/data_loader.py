import os

def get_project_root() -> str:
    """Tự động xác định thư mục gốc của dự án."""
    # File này nằm ở: Project_Root/Phase_1/loaders/data_loader.py
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_file_dir, "../../"))

def load_disease_knowledge(file_path: str) -> str:
    """Đọc file văn bản (.txt) chứa kiến thức bệnh lý lâm sàng."""
    if not os.path.isabs(file_path):
        file_path = os.path.join(get_project_root(), file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[LỖI DATA] Không tìm thấy file kiến thức bệnh: {file_path}")

    try:
        # Bắt buộc encoding='utf-8' để đọc đúng dấu Tiếng Việt
        with open(file_path, 'r', encoding='utf-8') as file:
            knowledge_data = file.read().strip()
            return knowledge_data
    except Exception as exc:
        raise RuntimeError(f"[LỖI DATA] Lỗi khi đọc file text: {exc}")

def get_image_path(image_path: str) -> str:
    """Kiểm tra sự tồn tại và trả về đường dẫn tuyệt đối của ảnh."""
    if not os.path.isabs(image_path):
        image_path = os.path.join(get_project_root(), image_path)
        
    abs_img_path = os.path.abspath(image_path)
    if not os.path.exists(abs_img_path):
        raise FileNotFoundError(f"[LỖI DATA] Không thấy ảnh tại: {abs_img_path}")
    return abs_img_path

def prepare_vlm_input(image_path: str, txt_path: str) -> tuple:
    """Gói gọn thao tác load ảnh và text thành 1 hàm duy nhất."""
    disease_knowledge = load_disease_knowledge(txt_path)
    abs_image_path = get_image_path(image_path)
    return abs_image_path, disease_knowledge

if __name__ == "__main__":
    # Test case sử dụng đường dẫn tương đối (Relative Path)
    test_img_rel = "Phase_1/assets/few_shot_image/Erythema_gyratum_repens.png"
    test_txt_rel = "Phase_1/assets/few_shot_knowledge/Erythema_gyratum_repens.txt" # Đã sửa tên file test cho gọn
    
    print(f"Project Root: {get_project_root()}")
    try:
        img_path, txt_content = prepare_vlm_input(test_img_rel, test_txt_rel)
        print(f"✅ Đã load kiến thức bệnh: {len(txt_content)} ký tự.")
        print(f"✅ Path ảnh sẵn sàng gửi đi: {img_path}")
    except Exception as e:
        print(f"❌ Lỗi thực thi: {e}")
        print("💡 Gợi ý: Hãy đảm bảo bạn đã tạo file ảnh và txt giả lập ở đúng đường dẫn để test.")