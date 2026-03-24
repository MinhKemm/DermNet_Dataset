import ollama
import json
from .vlm_engine import clean_and_parse_json

def build_judge_prompt(vlm1_data: dict, vlm2_data: dict, jaccard_score: float) -> str:
    """
    Tạo prompt nghiêm ngặt cho Judge LLM. Bám sát cấu trúc JSON extraction của bạn.
    """
    vlm1_str = json.dumps(vlm1_data, ensure_ascii=False, indent=2)
    vlm2_str = json.dumps(vlm2_data, ensure_ascii=False, indent=2)
    
    prompt = f"""Bạn là một chuyên gia AI Da liễu đánh giá dữ liệu (Data Annotator Judge).
Nhiệm vụ: Hợp nhất 2 bản trích xuất đặc điểm lâm sàng từ 2 mô hình thị giác máy tính khác nhau.

[THÔNG TIN ĐÁNH GIÁ]
- Độ tương đồng Jaccard hiện tại: {jaccard_score:.2f} 
(Lưu ý: Nếu < 0.85, 2 model đang mâu thuẫn. Hãy kiểm tra cực kỳ khắt khe).

[KẾT QUẢ MODEL 1 (Qwen)]
{vlm1_str}

[KẾT QUẢ MODEL 2 (LLaVA)]
{vlm2_str}

[NGUYÊN TẮC ZERO-HALLUCINATION BẮT BUỘC]
1. Sự Đồng Thuận: Chỉ giữ lại các đặc tính (Colour, Shape, Distribution, Characteristics...) mà CẢ HAI model đều đồng thuận hoặc bổ trợ logic cho nhau.
2. Loại Bỏ Mâu Thuẫn: Nếu Model 1 bảo "Đỏ", Model 2 bảo "Đen" -> Ghi "Không rõ" hoặc loại bỏ thuộc tính đó.
3. Không Bịa Đặt: TUYỆT ĐỐI không tự nghĩ ra thêm triệu chứng không có trong 2 bản báo cáo trên.

Hãy trả về ĐÚNG MỘT khối JSON duy nhất, giữ nguyên cấu trúc các key (Category, Lesion_Type, Colour, Shape, Distribution, Characteristics).
Không in thêm bất kỳ dòng text giải thích nào khác ngoài JSON.
"""
    return prompt


def run_judge(model_id: str, vlm1_data: dict, vlm2_data: dict, jaccard_score: float, temperature: float = 0.0) -> dict:
    """
    Hàm gọi Llama 3.2 để làm quan tòa phân xử và hợp nhất 2 JSON.
    
    Args:
        model_id (str): VD 'llama3.2:3b'.
        vlm1_data (dict): JSON từ Qwen.
        vlm2_data (dict): JSON từ LLaVA.
        jaccard_score (float): Điểm tương đồng.
        temperature (float): Nên để 0.0 để đảm bảo tính nhất quán (Deterministic).
    """
    print(f"[JUDGE - {model_id}] Bắt đầu phân xử. Jaccard Score: {jaccard_score:.2f}")
    
    # 1. CƠ CHẾ FALLBACK (Xử lý khi 1 trong 2 VLM bị sập)
    has_err_1 = "error" in vlm1_data
    has_err_2 = "error" in vlm2_data
    
    if has_err_1 and not has_err_2:
        print("[!] VLM1 lỗi, Judge tự động lấy kết quả VLM2.")
        vlm2_data['_metadata']['judge_action'] = "Fallback to VLM2"
        return vlm2_data
    elif has_err_2 and not has_err_1:
        print("[!] VLM2 lỗi, Judge tự động lấy kết quả VLM1.")
        vlm1_data['_metadata']['judge_action'] = "Fallback to VLM1"
        return vlm1_data
    elif has_err_1 and has_err_2:
        print("[!] Cả 2 VLM đều lỗi. Bỏ qua case này.")
        return {"error": "Cả 2 VLM đều failed."}

    # 2. CHẠY JUDGE (Khi cả 2 VLM đều thành công)
    prompt = build_judge_prompt(vlm1_data, vlm2_data, jaccard_score)
    
    try:
        # Llama 3.2 chỉ xử lý Text, không có parameter 'images'
        response = ollama.chat(
            model=model_id,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            format='json',
            options={'temperature': temperature}
        )
        
        raw_output = response.get('message', {}).get('content', '')
        result_dict = clean_and_parse_json(raw_output)
        
        result_dict['_metadata'] = {
            "model": model_id,
            "status": "success_consensus",
            "jaccard_input_score": jaccard_score
        }
        return result_dict
        
    except Exception as e:
        print(f"[-] [LỖI JUDGE - {model_id}]: {e}")
        return {"error": "Judge failed to consensus", "details": str(e)}