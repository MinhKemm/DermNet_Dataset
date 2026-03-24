import os
import sys
import time

# 1. TỰ ĐỘNG XÁC ĐỊNH ĐƯỜNG DẪN GỐC (PROJECT ROOT)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import các module thành phần
try:
    from Phase_1.loaders.config_loader import get_settings, get_prompts
    from Phase_1.loaders.data_loader import prepare_vlm_input
    from Phase_1.core.vlm_engine import run_vlm
    from Phase_1.core.judge_engine import run_judge
    from Phase_1.utils.metrics import calculate_jaccard
    from Phase_1.utils.json_handler import save_json
except ImportError as e:
    print(f"[LỖI IMPORT] Không thể tìm thấy các module thành phần: {e}")
    sys.exit(1)

def build_vlm_prompt(system_instruction: str, cot_steps: str, user_template: str, disease_text: str) -> str:
    formatted_user = user_template.replace("{disease_knowledge}", disease_text)
    return f"{system_instruction}\n\n[HƯỚNG DẪN TƯ DUY]\n{cot_steps}\n\n{formatted_user}"

def process_single_case(image_path: str, disease_txt_path: str, case_id: str):
    """Thực thi pipeline cho 1 case."""
    print(f"\n{'='*50}\nBẮT ĐẦU XỬ LÝ: {case_id}\n{'='*50}")
    start_time = time.time()
    
    settings = get_settings()
    prompts_data = get_prompts()
    
    # Đảm bảo các folder output tồn tại (Tự tạo nếu thiếu)
    inter_dir = os.path.join(project_root, settings['paths']['data_intermediate'])
    final_dir = os.path.join(project_root, settings['paths']['data_consensus'])
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # 2. Chuẩn bị đầu vào
    abs_img_path, disease_text = prepare_vlm_input(image_path, disease_txt_path)
    full_prompt = build_vlm_prompt(
        prompts_data['phase1']['system_instruction'],
        prompts_data['phase1']['cot_steps'],
        prompts_data['phase1']['user_template'],
        disease_text
    )
    
    # --- STEP 1: VLM INFERENCE ---
    vlm1_id = settings['models']['vlm_1']['model_id']
    vlm2_id = settings['models']['vlm_2']['model_id']
    
    json_vlm1 = run_vlm(vlm1_id, abs_img_path, full_prompt, settings['models']['vlm_1']['temperature'])
    json_vlm2 = run_vlm(vlm2_id, abs_img_path, full_prompt, settings['models']['vlm_2']['temperature'])
    
    save_json(json_vlm1, os.path.join(inter_dir, f"{case_id}_{vlm1_id.replace(':', '_')}.json"))
    save_json(json_vlm2, os.path.join(inter_dir, f"{case_id}_{vlm2_id.replace(':', '_')}.json"))

    # --- STEP 2: METRICS ---
    j_score = calculate_jaccard(json_vlm1, json_vlm2)
    print(f"-> Jaccard Score: {j_score:.4f}")

    # --- STEP 3: JUDGE ---
    judge_id = settings['models']['judge_llm']['model_id']
    json_final = run_judge(judge_id, json_vlm1, json_vlm2, j_score, settings['models']['judge_llm']['temperature'])
    
    save_json(json_final, os.path.join(final_dir, f"{case_id}_consensus.json"))
    
    print(f"[HOÀN THÀNH] Case {case_id} trong {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    settings = get_settings()
    
    # Lấy đường dẫn test_demo linh hoạt
    # settings['paths']['data_raw'] nên là "Phase_1/test_demo"
    test_base_dir = os.path.join(project_root, settings['paths']['data_raw'])
    
    if not os.path.exists(test_base_dir):
        print(f"❌ Không tìm thấy folder test: {test_base_dir}")
        sys.exit(1)

    # Duyệt qua các folder con (test_01, test_02,...)
    sub_folders = sorted([f for f in os.listdir(test_base_dir) if os.path.isdir(os.path.join(test_base_dir, f))])
    
    for folder_name in sub_folders:
        case_path = os.path.join(test_base_dir, folder_name)
        
        img_file = None
        txt_file = None
        
        # Tự động tìm file ảnh và text trong folder case
        for f in os.listdir(case_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_file = os.path.join(case_path, f)
            elif f.lower().endswith('.txt'):
                txt_file = os.path.join(case_path, f)
        
        if img_file and txt_file:
            process_single_case(img_file, txt_file, folder_name)
        else:
            print(f"⚠️ Folder {folder_name} thiếu file ảnh hoặc mô tả bệnh.")

    print("\n✅ TẤT CẢ CASE TRONG test_demo ĐÃ ĐƯỢC XỬ LÝ XONG.")