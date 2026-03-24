import ollama
import json 
import re
import os

def clean_and_parse_json(raw_txt: str) -> dict:
    """
    Hàm dọn dẹp các ký tự thừa và parse chuỗi thành Python Dictionary.
    Sửa lỗi Regex để lấy được JSON phức tạp/lồng nhau.
    """
    # Bước 1: Thử parse trực tiếp (tốc độ nhanh nhất)
    try:
        return json.loads(raw_txt.strip())
    except json.JSONDecodeError:
        pass

    # Bước 2: Dùng Regex "Greedy" để bắt từ dấu { đầu tiên đến dấu } cuối cùng
    # Thêm dấu () để tạo group 1
    match = re.search(r'(\{.*\})', raw_txt, re.DOTALL)
    
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Nếu vẫn lỗi, có thể do model sinh JSON thiếu dấu phẩy, v.v.
            raise ValueError(f"JSON tìm thấy không hợp lệ. Lỗi: {e}")
    
    raise ValueError(f"Không tìm thấy cấu trúc {{...}} trong output của model.")

def run_vlm(model_id: str, image_path: str, prompt: str, temperature: float = 0.1) -> dict:
    """
    Hàm gọi Vision Model qua Ollama.
    """
    if not os.path.exists(image_path):
        # Đối với Pipeline chạy hàng loạt, nên return dict lỗi thay vì raise để không dừng cả chương trình
        return {"error": f"FileNotFound: {image_path}"}

    raw_output = "" # Khởi tạo để tránh lỗi UnboundLocalError
    try:
        response = ollama.chat(
            model=model_id,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }],
            format='json',
            options={
                'temperature': temperature,
                'num_predict': 1024
            }
        )

        raw_output = response.get('message', {}).get('content', '')
        
        # Parse JSON
        result_dict = clean_and_parse_json(raw_output)
        
        # Thêm Metadata để phục vụ nghiên cứu (Paper cần tính minh bạch)
        result_dict['_metadata'] = {
            "model": model_id,
            "status": "success",
            "image": os.path.basename(image_path)
        }
        return result_dict

    except ValueError as ve:
        print(f"[-] [LỖI JSON - {model_id}]: {ve}")
        return {
            "error": "ParseError", 
            "raw_output": raw_output, 
            "_metadata": {"model": model_id, "status": "failed"}
        }
    except Exception as e:
        print(f"[-] [LỖI HỆ THỐNG - {model_id}]: {e}")
        return {
            "error": str(e), 
            "_metadata": {"model": model_id, "status": "failed"}
        }

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test case 1: JSON lồng nhau và có rác xung quanh
    test_raw = """
    Dưới đây là kết quả phân tích:
    {
      "benh": "Cham",
      "chi_tiet": {
         "mau": "do",
         "vay": "co"
      }
    }
    """
    try:
        print("Test Parse Thành Công:")
        print(json.dumps(clean_and_parse_json(test_raw), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Test Parse Thất Bại: {e}")