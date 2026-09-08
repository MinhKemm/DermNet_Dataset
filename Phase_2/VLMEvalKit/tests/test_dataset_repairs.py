import ast
import csv
import json
from pathlib import Path
import unittest
import unicodedata

KIT = Path(__file__).resolve().parents[1]


class RepairTests(unittest.TestCase):
    def test_all_dataset_images_exist(self):
        images = KIT.parents[1] / 'dermnet-output/images'
        normalize = lambda s: unicodedata.normalize('NFC', s).casefold()
        available = {normalize(p.relative_to(images).as_posix()) for p in images.rglob('*') if p.is_file()}
        missing = []
        for path in (KIT / 'LMUData').glob('DermNet_*_??.tsv'):
            with path.open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream, delimiter='\t'):
                    relative = row['image_path'].replace('\\', '/').split('/images/')[-1]
                    if normalize(relative) not in available:
                        missing.append((path.name, row['index']))
        self.assertEqual([], missing)

    def test_repair_audit_matches_current_questions(self):
        audit = json.loads((KIT / 'scripts/dataset_repairs.json').read_text(encoding='utf-8'))
        self.assertEqual(140, len(audit))
        for dataset in {item['dataset'] for item in audit}:
            with (KIT / f'LMUData/{dataset}.tsv').open(encoding='utf-8', newline='') as stream:
                rows = {int(row['index']): row for row in csv.DictReader(stream, delimiter='\t')}
            for item in audit:
                if item['dataset'] == dataset:
                    self.assertEqual(item['after'], rows[item['index']]['question'])

    def test_invalid_predictions_are_not_reused(self):
        tree = ast.parse((KIT / 'vlmeval/inference.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_valid_cached_prediction')
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<cache>', 'exec'), scope)
        valid = scope['_valid_cached_prediction']
        for value in (None, '', ' ', float('nan'), '<NA>', 'Failed to obtain answer.', 'SKIP: Image not found', {'prediction': ''}):
            self.assertFalse(valid(value), repr(value))
        self.assertTrue(valid({'prediction': 'Có'}))
