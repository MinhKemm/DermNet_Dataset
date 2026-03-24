import os 
import base64


def load_disease_knowledge(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file kiến thức của bệnh: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            knowledge_data = file.read().strip()
            return knowledge_data
    except Exception as exc:
        raise RuntimeError(f"Lỗi khi đọc file kiến thức của bệnh: {exc}")
        return ""


def get_image_path(image_path: str) -> str:
    """Kiểm tra và trả về đường dẫn tuyệt đối của ảnh."""
    abs_img_path = os.path.abspath(image_path)
    if not os.path.exists(abs_img_path):
        raise FileNotFoundError(f"[LỖI] Không thấy ảnh tại: {abs_img_path}")
    return abs_img_path

def prepare_vlm_input(image_path: str, txt_path: str) -> tuple:
    disease_knowledge = load_disease_knowledge(txt_path)
    image = get_image_path(image_path)

    return image, disease_knowledge


if __name__ == "__main__":
    # Thay bằng đường dẫn trong folder assets dùng debug
    test_img = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_1/assets/few_shot_image/Erythema_gyratum_repens.png"
    test_txt = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_1/assets/few_shot_knowledge/Toàn bộ nội dung - Erythema gyratum repens.txt"
    
    if os.path.exists(test_img) and os.path.exists(test_txt):
        img_path, txt_content = prepare_vlm_input(test_img, test_txt)
        print(f"Đã load kiến thức bệnh: {len(txt_content)} ký tự.")
        print(f"Path ảnh gửi đi: {img_path}")
    else:
        print("Vui lòng kiểm tra lại đường dẫn test.")


