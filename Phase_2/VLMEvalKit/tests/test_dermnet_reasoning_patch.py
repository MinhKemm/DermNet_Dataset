import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "dermnet_reasoning_patch.py"


def load_patch_module():
    spec = importlib.util.spec_from_file_location("dermnet_reasoning_patch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DermNetReasoningPatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = load_patch_module()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.base = self.root / "base.tsv"
        self.original = self.root / "original.tsv"
        self.mini = self.root / "reasoning.tsv"
        self.patch_result = self.root / "patch.tsv"
        self.backups = self.root / "backups"
        pd.DataFrame(
            [
                {"index": 1, "question": "old", "answer": "old", "category": "Diagnosis", "type": "Short_answer"},
                {"index": 2, "question": "new question", "answer": "Có", "category": "Lesion_Reasoning", "type": "Judgement"},
            ]
        ).to_csv(self.base, sep="\t", index=False)
        pd.DataFrame(
            [
                {"index": 1, "question": "old", "answer": "old", "category": "Diagnosis", "type": "Short_answer", "prediction": "keep"},
                {"index": 2, "question": "stale", "answer": "stale", "category": "Lesion_Reasoning", "type": "Judgement", "prediction": "wrong"},
            ]
        ).to_csv(self.original, sep="\t", index=False)

        for path in (self.base, self.original):
            frame = pd.read_csv(path, sep="\t")
            frame["image_path"] = ["/images/disease/a.jpg", "/images/disease/b.jpg"]
            frame.to_csv(path, sep="\t", index=False)

    def patch_row(self, prediction):
        return {"index": 2, "prediction": prediction, "image_path": "/images/disease/b.jpg",
                "question": "new question", "answer": "Có", "category": "Lesion_Reasoning", "type": "Judgement"}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepare_extracts_reasoning_without_modifying_original_result(self):
        before = self.original.read_bytes()

        count = self.patch.prepare_reasoning_dataset(self.base, self.mini)

        self.assertEqual(1, count)
        self.assertEqual(before, self.original.read_bytes())
        self.assertEqual([2], pd.read_csv(self.mini, sep="\t")["index"].tolist())

    def test_audited_question_repair_is_rerun_and_merged(self):
        item = json.loads(MODULE_PATH.with_name('dataset_repairs.json').read_text(encoding='utf-8'))[0]
        base = pd.DataFrame([{'index': item['index'], 'question': item['after'],
                              'answer': 'Có', 'category': 'Diagnosis', 'type': 'Judgement',
                              'image_path': item['image_path']}])
        original = base.copy()
        original['question'] = item['before']
        original['prediction'] = 'old'
        base.to_csv(self.base, sep='\t', index=False)
        original.to_csv(self.original, sep='\t', index=False)
        self.assertEqual(1, self.patch.prepare_reasoning_dataset(self.base, self.mini, self.original))
        result = pd.read_csv(self.mini, sep='\t')
        result['prediction'] = 'Có'
        result.to_csv(self.patch_result, sep='\t', index=False)
        self.patch.merge_reasoning_result(self.base, self.original, self.patch_result, self.backups)
        merged = pd.read_csv(self.original, sep='\t')
        self.assertEqual(item['after'], merged.iloc[0].question)
        self.assertEqual('Có', merged.iloc[0].prediction)

    def test_merge_updates_only_reasoning_and_creates_backup(self):
        pd.DataFrame([self.patch_row("Có")]).to_csv(
            self.patch_result, sep="\t", index=False
        )

        backup = self.patch.merge_reasoning_result(
            self.base, self.original, self.patch_result, self.backups
        )

        merged = pd.read_csv(self.original, sep="\t")
        self.assertEqual("keep", merged.loc[merged["index"] == 1, "prediction"].item())
        self.assertEqual("Có", merged.loc[merged["index"] == 2, "prediction"].item())
        self.assertEqual("new question", merged.loc[merged["index"] == 2, "question"].item())
        self.assertTrue(backup.is_file())

    def test_merge_rejects_incomplete_patch_without_modifying_original(self):
        pd.DataFrame(columns=list(self.patch_row("Có"))).to_csv(
            self.patch_result, sep="\t", index=False
        )
        before = self.original.read_bytes()

        with self.assertRaisesRegex(ValueError, "missing reasoning predictions|missing dataset indices"):
            self.patch.merge_reasoning_result(
                self.base, self.original, self.patch_result, self.backups
            )

        self.assertEqual(before, self.original.read_bytes())

    def test_merge_rejects_blank_and_skipped_predictions(self):
        before = self.original.read_bytes()
        for prediction in ("   ", "SKIP: Image not found", "Failed to obtain answer"):
            with self.subTest(prediction=prediction):
                pd.DataFrame([self.patch_row(prediction)]).to_csv(
                    self.patch_result, sep="\t", index=False
                )
                with self.assertRaisesRegex(ValueError, "missing reasoning predictions|missing dataset indices"):
                    self.patch.merge_reasoning_result(
                        self.base, self.original, self.patch_result, self.backups
                    )
                self.assertEqual(before, self.original.read_bytes())

    def test_merge_excel_preserves_backup_and_clears_reasoning_score(self):
        original = self.root / "original.xlsx"
        frame = pd.read_csv(self.original, sep="\t")
        frame["score"] = [1, 0]
        frame.to_excel(original, index=False)
        before = original.read_bytes()
        pd.DataFrame([self.patch_row("Có")]).to_csv(
            self.patch_result, sep="\t", index=False
        )
        backup = self.patch.merge_reasoning_result(
            self.base, original, self.patch_result, self.backups
        )
        self.assertEqual(before, backup.read_bytes())
        merged = pd.read_excel(original).set_index("index")
        self.assertEqual("keep", merged.loc[1, "prediction"])
        self.assertEqual(1, merged.loc[1, "score"])
        self.assertEqual("Có", merged.loc[2, "prediction"])
        self.assertTrue(pd.isna(merged.loc[2, "score"]))

    def test_separate_merge_keeps_unmodified_source_excel(self):
        source = self.root / "untouched.xlsx"
        output = self.root / "merged" / "result.xlsx"
        pd.read_csv(self.original, sep="\t").to_excel(source, index=False)
        before = source.read_bytes()
        pd.DataFrame([self.patch_row("Có")]).to_csv(
            self.patch_result, sep="\t", index=False
        )
        self.patch.merge_reasoning_result(
            self.base, source, self.patch_result, self.backups, output
        )
        self.assertEqual(before, source.read_bytes())
        merged = pd.read_excel(output).set_index("index")
        self.assertEqual("Có", merged.loc[2, "prediction"])
        self.assertEqual("keep", merged.loc[1, "prediction"])

    def test_same_indices_with_different_images_are_rejected(self):
        frame = pd.read_csv(self.original, sep="\t")
        frame.loc[frame["index"] == 2, "image_path"] = "/images/other/wrong.jpg"
        frame.to_csv(self.original, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "image mismatch"):
            self.patch.validate_source(self.base, self.original)

    def test_nonreasoning_question_changes_are_rejected(self):
        frame = pd.read_csv(self.original, sep="\t")
        frame.loc[frame["index"] == 1, "question"] = "unrelated question"
        frame.to_csv(self.original, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "question mismatch"):
            self.patch.validate_source(self.base, self.original)

    def test_windows_and_linux_image_prefixes_match(self):
        frame = pd.read_csv(self.original, sep="\t")
        frame["image_path"] = [r"D:\data\images\disease\a.jpg", r"D:\data\images\disease\b.jpg"]
        frame.to_csv(self.original, sep="\t", index=False)
        self.assertEqual(2, self.patch.validate_source(self.base, self.original))


if __name__ == "__main__":
    unittest.main()
