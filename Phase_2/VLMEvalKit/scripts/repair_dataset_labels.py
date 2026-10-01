"""Deterministic repair using existing canonical labels; retain an audit trail."""
import csv
import json
from pathlib import Path
import re
import unicodedata


def main():
    kit = Path(__file__).resolve().parents[1]
    root = kit.parent.parent
    canonical = root / 'final_canonical_vi'
    lookup = {unicodedata.normalize('NFC', str(p.relative_to(canonical))).casefold(): p
              for p in canonical.rglob('*.json')}
    audit_path = kit / 'scripts/dataset_repairs.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else []
    for split in ('Val', 'Test'):
        for language in ('VI', 'EN'):
            dataset = f'DermNet_{split}_{language}'
            path = kit / f'LMUData/{dataset}.tsv'
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream, delimiter='\t')
                columns, rows = reader.fieldnames, list(reader)
            changed = False
            for row in rows:
                before = row['question']
                source = ''
                pattern = r'chẩn đoán (?:Không|Có)(?=[.?])' if language == 'VI' else r'diagnosis (?:No|Yes)(?=[.?])'
                if row['type'] == 'Judgement' and re.search(pattern, before):
                    if row['answer'] != ('Có' if language == 'VI' else 'Yes'):
                        raise ValueError('Cannot repair a negative judgement by substituting the canonical label')
                    relative = Path(row['image_path'].replace('\\', '/').split('/images/')[-1]).with_suffix('.json')
                    source_path = lookup[unicodedata.normalize('NFC', str(relative)).casefold()]
                    label = json.loads(source_path.read_text(encoding='utf-8'))['TRICH_XUAT_JSON']['Danh_muc_benh']
                    if not isinstance(label, str) or not label.strip():
                        raise ValueError(f'Missing label in {source_path}')
                    prefix = 'chẩn đoán ' if language == 'VI' else 'diagnosis '
                    row['question'] = re.sub(pattern, lambda m: prefix + label, before)
                    source = str(source_path.relative_to(root)).replace('\\', '/')
                elif row['type'] == 'Multi_choice':
                    row['question'] = re.sub(r'(?m)^([A-D]) (?=\S)', r'\1. ', before)
                if row['question'] != before:
                    changed = True
                    audit.append({'dataset': dataset, 'index': int(row['index']),
                                  'image_path': row['image_path'], 'before': before,
                                  'after': row['question'], 'canonical_source': source})
            if changed:
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, columns, delimiter='\t', lineterminator='\r\n')
                    writer.writeheader()
                    writer.writerows(rows)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Audited question repairs: {len(audit)}')


if __name__ == '__main__':
    main()
