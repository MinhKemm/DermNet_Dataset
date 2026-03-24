import yaml
import os
import sys

def load_yaml_config(file_path: str) -> dict:
    # --- PHẦN SỬA ĐỔI ĐƯỜNG DẪN ---
    # Lấy đường dẫn của thư mục chứa file config_loader.py hiện tại
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tìm về thư mục gốc của dự án (Giả sử file này nằm trong Phase_1/loaders/)
    # Ta cần đi ngược lên 2 cấp để ra ngoài Phase_1
    project_root = os.path.abspath(os.path.join(current_file_dir, "../../"))
    
    # Ghép với file_path tương đối truyền vào
    abs_file_path = os.path.join(project_root, file_path)
    # ------------------------------

    if not os.path.exists(abs_file_path):
        raise FileNotFoundError(f"File cấu hình không tồn tại: {abs_file_path}")

    try:
        with open(abs_file_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
            return config_data
    except yaml.YAMLError as e: # Sửa lỗi typo YAMLError -> yaml.YAMLError
        raise ValueError(f"Lỗi khi đọc file YAML: {e}")
    except Exception as exc:
        raise RuntimeError(f"Lỗi không xác định khi đọc file YAML: {exc}")

def get_settings():
    # Chỉ cần truyền đường dẫn tính từ gốc dự án
    return load_yaml_config('Phase_1/config/settings.yaml')

def get_prompts():
    return load_yaml_config('Phase_1/config/prompts.yaml')

if __name__ == "__main__":
    try:
        settings = get_settings()
        print(f"Cấu trúc settings: {type(settings)}")
        print("✅ Đã load thành công settings")
        print(f"Jaccard Threshold: {settings['pipeline']['jaccard_threshold']}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")