import csv
from pathlib import Path
import re
import unittest


DATA_DIR = Path(__file__).parents[1] / "LMUData"
DATASETS = (
    "DermNet_Val_VI.tsv",
    "DermNet_Test_VI.tsv",
    "DermNet_Val_EN.tsv",
    "DermNet_Test_EN.tsv",
)


class DermNetDatasetContractTest(unittest.TestCase):
    def test_closed_answer_types_match_the_required_output_format(self):
        errors = []
        for dataset_name in DATASETS:
            language = "en" if dataset_name.lower().endswith("_en.tsv") else "vi"
            valid_judgements = {"Yes", "No"} if language == "en" else {"Có", "Không"}
            with (DATA_DIR / dataset_name).open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    answer = row["answer"].strip()
                    if row["type"] == "Judgement" and answer not in valid_judgements:
                        errors.append((dataset_name, row["index"], row["type"], answer))
                    if row["type"] == "Multi_choice" and not re.fullmatch(
                        r"[A-D]+", answer
                    ):
                        errors.append((dataset_name, row["index"], row["type"], answer))

        self.assertEqual([], errors[:20], f"Found {len(errors)} invalid closed answers")


if __name__ == "__main__":
    unittest.main()
