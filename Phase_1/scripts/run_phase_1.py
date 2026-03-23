# scripts/01_run_phase1.py
import os
import sys
import yaml
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loaders.dermnet_loader import DermNetLoader
from src.core.prompt_builder import PromptBuilder
from src.models.ollama_client import OllamaClient # Import duy nhất Client này
from src.utils.json_parser import extract_json_from_text

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def main():
    print("🚀 BẮT ĐẦU CHẠY PHASE 1 (LOCAL M3 DEBUG)")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings = load_yaml(os.path.join(project_root, "configs", "settings.yaml"))
    prompts_cfg = load_yaml(os.path.join(project_root, "configs", "prompts.yaml"))
    
    data_raw_path = os.path.join(project_root, settings['paths']['data_raw'])
    out_dir = os.path.join(project_root, settings['paths']['data_intermediate'])
    assets_dir = os.path.join(project_root, settings['paths']['prompt_assets'])
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase1_results.jsonl")

    loader = DermNetLoader(data_raw_path)
    samples = loader.get_all_samples()
    
    p_builder = PromptBuilder(prompts_cfg['phase1'], assets_dir)
    sys_prompt = p_builder.get_system_prompt()
    few_shot_msgs = p_builder.get_few_shot_messages()
    
    print("\n⏳ Khởi tạo VLM1 (Qwen) & VLM2 (LLaVA)...")
    vlm_1 = OllamaClient(settings['models']['vlm_1'])
    vlm_2 = OllamaClient(settings['models']['vlm_2'])

    with open(out_file, 'a', encoding='utf-8') as f_out:
        for sample in tqdm(samples, desc="Processing Images"):
            img_path = sample['image_path']
            user_prompt = p_builder.get_user_prompt(sample['disease_knowledge'], sample['clinical_description'])
            
            # --- Inference Local ---
            res_vlm1_raw = vlm_1.generate(img_path, sys_prompt, few_shot_msgs, user_prompt)
            res_vlm2_raw = vlm_2.generate(img_path, sys_prompt, few_shot_msgs, user_prompt)
            
            record = {
                "sample_id": sample['sample_id'],
                "disease_name": sample['disease_name'],
                "image_path": img_path,
                "vlm1_qwen_json": extract_json_from_text(res_vlm1_raw),
                "vlm2_llava_json": extract_json_from_text(res_vlm2_raw),
                "vlm1_raw": res_vlm1_raw,
                "vlm2_raw": res_vlm2_raw
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            
    print(f"✅ HOÀN TẤT PHASE 1. Dữ liệu: {out_file}")

if __name__ == "__main__":
    main()