import os
import json
import csv
from pathlib import Path

def export_tsv():
    dataset_root = Path("/Users/binhminh/Desktop/DermNet_Dataset")
    output_dir = dataset_root / "Phase_1" / "output" / "vqa"
    tsv_out = dataset_root / "Phase_1" / "output" / "dermnet_vqa_benchmark.tsv"
    
    if not output_dir.exists():
        print("Không tìm thấy thư mục vqa output.")
        return
        
    tsv_out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tsv_out, 'w', encoding='utf-8', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['image_path', 'category', 'type', 'question', 'answer'])
        
        count = 0
        for disease_dir in output_dir.iterdir():
            if not disease_dir.is_dir():
                continue
                
            for json_file in disease_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as jf:
                        items = json.load(jf)
                        for item in items:
                            writer.writerow([
                                item.get('image_path', ''),
                                item.get('category', ''),
                                item.get('type', ''),
                                item.get('question', '').replace('\n', '\\n'),
                                str(item.get('answer', '')).replace('\n', '\\n')
                            ])
                            count += 1
                except Exception as e:
                    print(f"Lỗi khi đọc {json_file}: {e}")
                    
    print(f"Đã xuất thành công {count} VQA pairs ra file TSV: {tsv_out}")

if __name__ == "__main__":
    export_tsv()
