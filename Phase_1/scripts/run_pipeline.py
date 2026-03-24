import os
import sys
import time

# Thêm đường dẫn gốc của dự án vào sys.path để Python tìm thấy package 'Phase_1'
# __file__ là đường dẫn của file hiện tại (Phase_1/scripts/run_pipeline.py)
# Thêm thư mục chứa 'Phase_1' vào đường dẫn tìm kiếm của Python
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Bây giờ các lệnh import từ Phase_1 sẽ hoạt động bình thường
try:
    from Phase_1.loaders.config_loader import get_settings, get_prompts
    from Phase_1.loaders.data_loader import prepare_vlm_input
    from Phase_1.core.vlm_engine import run_vlm
    from Phase_1.core.judge_engine import run_judge
    from Phase_1.utils.metrics import calculate_jaccard
    from Phase_1.utils.json_handler import save_json
except ImportError as e:
    print(f"[LỖI IMPORT] Không thể tìm thấy các module thành phần: {e}")
    print(f"Đường dẫn hiện tại của script: {current_dir}")
    print(f"Đường dẫn gốc dự án được xác định: {project_root}")
    sys.exit(1)

def build_vlm_prompt(system_instruction: str, cot_steps: str, user_template: str, disease_text: str) -> str:
    """Ghép nối prompt từ YAML và nội dung bệnh lý."""
    formatted_user = user_template.replace("{disease_knowledge}", disease_text)
    full_prompt = f"{system_instruction}\n\n[HƯỚNG DẪN TƯ DUY]\n{cot_steps}\n\n{formatted_user}"
    return full_prompt

def process_single_case(image_path: str, disease_txt_path: str, case_id: str):
    """
    Hàm thực thi toàn bộ pipeline cho 1 cặp (Ảnh + File Text bệnh).
    """
    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU XỬ LÝ CASE: {case_id}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 1. Load Cấu hình & Prompt
    settings = get_settings()
    prompts_data = get_prompts()
    
    p1_config = prompts_data['phase1']
    sys_inst = p1_config['system_instruction']
    cot = p1_config['cot_steps']
    user_tmpl = p1_config['user_template']
    
    # 2. Nạp Dữ liệu
    abs_img_path, disease_text = prepare_vlm_input(image_path, disease_txt_path)
    
    if not abs_img_path or not os.path.exists(abs_img_path):
        print(f"[!] Bỏ qua case {case_id} do lỗi đường dẫn ảnh.")
        return

    full_prompt = build_vlm_prompt(sys_inst, cot, user_tmpl, disease_text)
    
    # --- PHƯƠNG THỨC 1: VLM INFERENCE ---
    print("\n--- BƯỚC 1: TRÍCH XUẤT ĐẶC ĐIỂM (VLM) ---")
    vlm1_model = settings['models']['vlm_1']['model_id']
    vlm1_temp = settings['models']['vlm_1']['temperature']
    
    vlm2_model = settings['models']['vlm_2']['model_id']
    vlm2_temp = settings['models']['vlm_2']['temperature']
    
    json_vlm1 = run_vlm(vlm1_model, abs_img_path, full_prompt, vlm1_temp)
    json_vlm2 = run_vlm(vlm2_model, abs_img_path, full_prompt, vlm2_temp)
    
    # LƯU KIỂM SOÁT 1: Lưu kết quả trung gian
    inter_dir = settings['paths']['data_intermediate']
    save_json(json_vlm1, os.path.join(inter_dir, f"{case_id}_{vlm1_model.replace(':', '_')}.json"))
    save_json(json_vlm2, os.path.join(inter_dir, f"{case_id}_{vlm2_model.replace(':', '_')}.json"))

    # --- PHƯƠNG THỨC 2: ĐÁNH GIÁ (METRICS) ---
    print("\n--- BƯỚC 2: TÍNH TOÁN ĐỘ TƯƠNG ĐỒNG ---")
    j_score = calculate_jaccard(json_vlm1, json_vlm2)
    print(f"-> Jaccard Score: {j_score:.4f}")

    # --- PHƯƠNG THỨC 3: PHÂN XỬ (JUDGE) ---
    print("\n--- BƯỚC 3: LLM PHÂN XỬ (ZERO-HALLUCINATION) ---")
    judge_model = settings['models']['judge_llm']['model_id']
    judge_temp = settings['models']['judge_llm']['temperature']
    
    json_final = run_judge(judge_model, json_vlm1, json_vlm2, j_score, judge_temp)
    
    # LƯU KIỂM SOÁT 2: Lưu kết quả cuối cùng
    final_dir = settings['paths']['data_consensus']
    save_json(json_final, os.path.join(final_dir, f"{case_id}_consensus.json"))
    
    elapsed = time.time() - start_time
    print(f"\n[HOÀN THÀNH] Case {case_id} xử lý mất {elapsed:.2f} giây.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Test image và text
    test_image = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_1/assets/few_shot_image/Erythema_gyratum_repens.png"
    test_text = "/Users/binhminh/Desktop/DermNet_Dataset/Phase_1/assets/few_shot_knowledge/Toàn bộ nội dung - Erythema gyratum repens.txt"
    
    case_id = "Erythema_gyratum_repens"
    
    process_single_case(test_image, test_text, case_id)