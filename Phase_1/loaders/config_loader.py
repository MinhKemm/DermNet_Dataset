import yaml
import os

def get_project_root() -> str:
    """Tự động xác định thư mục gốc của dự án."""
    # File này nằm ở: Project_Root/Phase_1/loaders/config_loader.py
    # Đi ngược lên 2 cấp để ra thư mục gốc
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_file_dir, "../../"))

def load_yaml_config(file_path: str) -> dict:
    """Đọc và parse file YAML thành Dictionary."""
    project_root = get_project_root()
    
    # Nếu đường dẫn không phải tuyệt đối, ghép nó với project root
    if not os.path.isabs(file_path):
        abs_file_path = os.path.join(project_root, file_path)
    else:
        abs_file_path = file_path

    if not os.path.exists(abs_file_path):
        raise FileNotFoundError(f"[LỖI CONFIG] Không tìm thấy file cấu hình tại: {abs_file_path}")

    try:
        with open(abs_file_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
            return config_data
    except yaml.YAMLError as e:
        raise ValueError(f"[LỖI YAML] Định dạng file YAML không hợp lệ: {e}")
    except Exception as exc:
        raise RuntimeError(f"[LỖI HỆ THỐNG] Không thể đọc file YAML: {exc}")

def get_settings():
    """Lấy file cấu hình cài đặt chung."""
    return load_yaml_config('Phase_1/config/settings.yaml')

def get_prompts():
    """Lấy file cấu hình prompts cho VLM."""
    return load_yaml_config('Phase_1/config/prompts.yaml')

if __name__ == "__main__":
    print(f"Project Root: {get_project_root()}")
    try:
        settings = get_settings()
        print("✅ Đã load thành công settings.yaml")
        print(f"- Jaccard Threshold: {settings.get('pipeline', {}).get('jaccard_threshold', 'N/A')}")
        
        prompts = get_prompts()
        print("✅ Đã load thành công prompts.yaml")
        print(f"- Phase 1 System Prompt: {prompts['phase1_observation_qa']['system_instruction'][:50]}...")
    except Exception as e:
        print(f"❌ Lỗi Test Loader: {e}")