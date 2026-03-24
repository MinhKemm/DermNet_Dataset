import os 
import base64

def get_project_root():
    """Tự động xác định thư mục gốc của dự án."""
    # File này đang nằm ở: Phase_1/loaders/data_loader.py
    # Đi ngược lên 2 cấp để ra thư mục gốc DermNet_Dataset
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "../../"))
    return project_root

def load_disease_knowledge(file_path: str) -> str:
    # Nếu file_path truyền vào là đường dẫn tương đối, ghép nó với project root
    if not os.path.isabs(file_path):
        file_path = os.path.join(get_project_root(), file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file kiến thức của bệnh: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            knowledge_data = file.read().strip()
            return knowledge_data
    except Exception as exc:
        # Xóa dòng return "" sau raise vì code sẽ không bao giờ chạy đến đó
        raise RuntimeError(f"Lỗi khi đọc file kiến thức của bệnh: {exc}")

def get_image_path(image_path: str) -> str:
    """Kiểm tra và trả về đường dẫn tuyệt đối của ảnh."""
    # Nếu là đường dẫn tương đối, ghép với project root
    if not os.path.isabs(image_path):
        image_path = os.path.join(get_project_root(), image_path)
        
    abs_img_path = os.path.abspath(image_path)
    if not os.path.exists(abs_img_path):
        raise FileNotFoundError(f"[LỖI] Không thấy ảnh tại: {abs_img_path}")
    return abs_img_path

def prepare_vlm_input(image_path: str, txt_path: str) -> tuple:
    disease_knowledge = load_disease_knowledge(txt_path)
    image = get_image_path(image_path)
    return image, disease_knowledge

if __name__ == "__main__":
    # SỬ DỤNG ĐƯỜNG DẪN TƯƠNG ĐỐI (Tính từ gốc dự án)
    # Cách này giúp code chạy được trên cả Mac và Colab mà không cần sửa file
    test_img_rel = "Phase_1/assets/few_shot_image/Erythema_gyratum_repens.png"
    test_txt_rel = "Phase_1/assets/few_shot_knowledge/Toàn bộ nội dung - Erythema gyratum repens.txt"
    
    try:
        img_path, txt_content = prepare_vlm_input(test_img_rel, test_txt_rel)
        print(f"✅ Đã load kiến thức bệnh: {len(txt_content)} ký tự.")
        print(f"✅ Path ảnh gửi đi: {img_path}")
    except Exception as e:
        print(f"❌ Lỗi thực thi: {e}")
        print("Mẹo: Đảm bảo bạn đang đứng ở thư mục dự án hoặc cấu trúc folder đúng.")