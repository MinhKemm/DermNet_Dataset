# src/utils/json_parser.py
import json
import re

def extract_json_from_text(text: str) -> dict:
    """
    Dùng Regex để tìm và parse chuỗi JSON nằm giữa ký hiệu ```json và ``` 
    hoặc dấu ngoặc nhọn lớn nhất.
    """
    try:
        # Cố gắng tìm khối markdown JSON trước
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: Tìm ngoặc nhọn mở và đóng ngoài cùng
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("Không tìm thấy cấu trúc JSON trong văn bản.")
                
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Lỗi parse JSON: {e}")
        return {}
    except Exception as e:
        print(f"Lỗi không xác định khi parse JSON: {e}")
        return {}