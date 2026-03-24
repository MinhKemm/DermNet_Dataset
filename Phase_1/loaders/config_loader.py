import yaml
import os

def load_yaml_config(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File cấu hình không tồn tại:{file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
            return config_data
    except YAMLError as e:
        raise ValueERror(f"Lỗi khi đọc file YAML: {e}")
    except Exception as exc:
        raise RuntimeError(f"Lỗi không xác định khi đọc file YAML:{exc}")

def get_settings():
    return load_yaml_config('Phase_1/config/settings.yaml')

def get_prompts():
    return load_yaml_config('Phase_1/config/prompts.yaml')

if __name__ == "__main__":
    settings = get_settings()
    print(type(settings))
    print("Đã load thành công settings")
    print(f"Jaccard Theshold: {settings['pipeline']['jaccard_threshold']}")