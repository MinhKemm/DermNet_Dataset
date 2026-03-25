import os
import sys
import json
import time

# 1. TỰ ĐỘNG XÁC ĐỊNH ĐƯỜNG DẪN GỐC (PROJECT ROOT)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lùi lại 2 cấp từ thư mục scripts (scripts -> Phase_1 -> Project Root)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from Phase_1.loaders.config_loader import get_prompts
    from Phase_1.core.vlm_engine import VLMEngine
    from Phase_1.core.judge_engine import JudgeEngine
    from Phase_1.utils.metrics import calculate_jaccard
    from Phase_1.utils.json_handler import save_json_ordered
except ImportError as e:
    print(f"[LỖI IMPORT]: {e}. Vui lòng kiểm tra lại cấu trúc thư mục.")
    sys.exit(1)

# --- HELPER FUNCTIONS CHO PROMPT FEW-SHOT ---
def build_p1_prompt(p1_config):
    """Lắp ráp Prompt Phase 1 từ file YAML (bao gồm Few-shot)"""
    user_template = p1_config.get('user_template', '')
    examples = p1_config.get('few_shot_examples', [])
    
    if examples:
        user_template += "\n\n--- CÁC VÍ DỤ MINH HỌA ---\n"
        for ex_item in examples:
            for ex_key, ex_val in ex_item.items():
                img_name = os.path.basename(ex_val['image_name'])
                user_template += f"Ví dụ ({img_name}):\n{ex_val['expected_output'].strip()}\n\n"
        user_template += "--> BÂY GIỜ LÀ LƯỢT CỦA BẠN (Vui lòng chỉ trả lời 5 mục):"
        
    return user_template

def build_p2_prompt(p2_config, obs_output, disease_name, knowledge):
    """Lắp ráp Prompt Phase 2 từ file YAML (bao gồm Few-shot)"""
    user_template = p2_config.get('user_template', '')
    
    # Format template chính trước
    formatted_template = user_template.format(
        phase1_qa_output=obs_output,
        disease_name=disease_name,
        disease_knowledge=knowledge
    )
    
    examples = p2_config.get('few_shot_examples', [])
    if examples:
        formatted_template += "\n\n--- CÁC VÍ DỤ MINH HỌA ---\n"
        for ex_item in examples:
            for ex_key, ex_val in ex_item.items():
                formatted_template += f"Ví dụ ({ex_val['disease_name']}):\n"
                formatted_template += f"QA Đầu vào:\n{ex_val['phase1_qa_output'].strip()}\n\n"
                formatted_template += f"JSON Đầu ra:\n{ex_val['expected_json'].strip()}\n\n"
        formatted_template += f"--> BÂY GIỜ LÀ LƯỢT CỦA BẠN TRÍCH XUẤT JSON CHO BỆNH {disease_name}:"
        
    return formatted_template


def process_single_case(image_path: str, disease_txt_path: str, folder_name: str, prompts_config: dict):
    # Trích xuất tên bệnh THỰC SỰ từ tên file .txt (Bỏ chữ "Toàn bộ nội dung - " và ".txt")
    txt_filename = os.path.basename(disease_txt_path)
    real_disease_name = txt_filename.replace("Toàn bộ nội dung -", "").replace(".txt", "").strip()
    
    print(f"\n{'='*70}")
    print(f"🚀 BẮT ĐẦU XỬ LÝ FOLDER: {folder_name}")
    print(f"🧬 TÊN BỆNH LÂM SÀNG: {real_disease_name}")
    print(f"{'='*70}")
    
    # 1. Đọc kiến thức bệnh
    with open(disease_txt_path, 'r', encoding='utf-8') as f:
        disease_knowledge = f.read().strip()
    
    vlm_results = [] 
    models_to_test = [
        "Qwen/Qwen2-VL-7B-Instruct",
        "OpenGVLab/InternVL2-8B"
    ]

    vlm_engine = VLMEngine()

    # =========================================================
    # BƯỚC 1 & 2: VLM QUAN SÁT (PHASE 1) & MAPPING (PHASE 2)
    # =========================================================
    for m_id in models_to_test:
        print(f"\n[{m_id}] >>> ĐANG NẠP MODEL VÀO GPU...")
        try:
            vlm_engine.load_model(m_id)
            
            # --- PHASE 1 ---
            print(f"[{m_id}] Thực thi Phase 1 (Quan sát)...")
            sys_p1 = prompts_config["phase1_observation_qa"]["system_instruction"]
            usr_p1 = build_p1_prompt(prompts_config["phase1_observation_qa"])
            
            p1_res = vlm_engine.call_vlm(system_prompt=sys_p1, user_prompt=usr_p1, image_path=image_path)
            print(f"\n--- KẾT QUẢ PHASE 1 ({m_id}) ---\n{p1_res}\n-----------------------------------")
            
            # --- PHASE 2 ---
            print(f"[{m_id}] Thực thi Phase 2 (Chuẩn hóa JSON)...")
            sys_p2 = prompts_config["phase2_json_standardization"]["system_instruction"]
            usr_p2 = build_p2_prompt(prompts_config["phase2_json_standardization"], p1_res, real_disease_name, disease_knowledge)
            
            p2_res_raw = vlm_engine.call_vlm(system_prompt=sys_p2, user_prompt=usr_p2)
            p2_json = vlm_engine.extract_json(p2_res_raw)
            
            print(f"\n--- KẾT QUẢ PHASE 2 ({m_id}) ---")
            print(json.dumps(p2_json, indent=2, ensure_ascii=False))
            vlm_results.append(p2_json)
            
        except Exception as e:
            print(f"[-] LỖI TRONG QUÁ TRÌNH CHẠY {m_id}: {e}")
            vlm_results.append({"error": str(e), "Category": real_disease_name})
            
        finally:
            print(f"[{m_id}] <<< ĐANG XẢ VRAM...")
            vlm_engine.flush_memory()

    # =========================================================
    # BƯỚC 3: TÍNH JACCARD & JUDGE PHÂN XỬ
    # =========================================================
    if len(vlm_results) == 2:
        print(f"\n{'='*70}\n⚖️ BẮT ĐẦU PHÂN XỬ (JUDGE CONSENSUS)\n{'='*70}")
        
        data1 = vlm_results[0].get("JSON_EXTRACTION", vlm_results[0])
        data2 = vlm_results[1].get("JSON_EXTRACTION", vlm_results[1])
        
        j_score = calculate_jaccard(data1, data2)
        print(f"[METRIC] Jaccard Similarity Score: {j_score:.4f}")
        
        judge_engine = JudgeEngine(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
        try:
            print("[JUDGE] >>> ĐANG NẠP MODEL PHÂN XỬ...")
            judge_engine.load_model()
            
            final_json = judge_engine.run_judge(vlm_results[0], vlm_results[1], real_disease_name)
            
            print(f"\n--- KẾT QUẢ HỢP NHẤT CUỐI CÙNG (GOLD STANDARD) ---")
            print(json.dumps(final_json, indent=2, ensure_ascii=False))
            
            data_to_save = final_json.get("JSON_EXTRACTION", final_json)
            
            # Lưu file vào thư mục output (Dùng folder_name là test_01 để dễ map lại với ảnh gốc)
            final_path = os.path.join(project_root, "Phase_1", "output", "consensus", f"{folder_name}_final.json")
            save_json_ordered(data_to_save, final_path)
            
        except Exception as e:
            print(f"[-] LỖI CHẠY JUDGE: {e}")
        finally:
            print("[JUDGE] <<< ĐANG XẢ VRAM...")
            judge_engine.flush_memory()

if __name__ == "__main__":
    start_time = time.time()
    
    # Nạp toàn bộ Prompts từ file YAML
    try:
        PROMPTS_CONFIG = get_prompts()
        print("✅ Đã nạp thành công bộ Prompts từ YAML.")
    except Exception as e:
        print(f"[-] Lỗi nạp Prompts: {e}")
        sys.exit(1)
    
    # ---------------------------------------------------------
    # DỰA VÀO ẢNH CHỤP MÀN HÌNH, THƯ MỤC CỦA BẠN LÀ test_01
    # ---------------------------------------------------------
    TEST_FOLDER_NAME = "test_01" 
    TEST_DIR = os.path.join(project_root, "Phase_1", "test_demo", TEST_FOLDER_NAME)
    
    print(f"\nThư mục làm việc hiện tại: {os.getcwd()}")
    print(f"Đường dẫn Test Directory: {TEST_DIR}")
    
    if os.path.exists(TEST_DIR):
        img_file = next((os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), None)
        txt_file = next((os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith('.txt')), None)
        
        if img_file and txt_file:
            # Truyền prompts_config vào hàm
            process_single_case(
                image_path=img_file, 
                disease_txt_path=txt_file, 
                folder_name=TEST_FOLDER_NAME,
                prompts_config=PROMPTS_CONFIG
            )
        else:
            print(f"[-] LỖI: Thư mục '{TEST_DIR}' thiếu file ảnh hoặc file .txt!")
    else:
        print(f"[-] LỖI: Không tìm thấy thư mục '{TEST_DIR}'")
        
    print(f"\n⏱️ Tổng thời gian chạy pipeline: {(time.time() - start_time)/60:.2f} phút")