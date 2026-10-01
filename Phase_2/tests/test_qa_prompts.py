"""Test the consuming prompt builder, not the wording of the source file."""

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def api():
    # A missing implementation is an explicit failure during the initial RED run.
    try:
        return importlib.import_module("Phase_2.qa_prompts")
    except ModuleNotFoundError as exc:
        raise AssertionError("The standard prompt builder is not implemented") from exc


def context():
    return {
        "image_id": "example-001",
        "image_path": "images/example.png",
        "taxonomy": {
            "version": "fixture-reviewed-1",
            "fields": {
                "Category": ["Sẩn", "Dát", "Mảng", "Vảy da"],
                "Location": ["Mu bàn tay", "Lòng bàn tay", "Má", "Trán"],
                "Color": ["Màu đỏ", "Màu trắng", "Màu đen", "Màu tím"],
                "Shape": ["Hình tròn", "Hình bầu dục", "Dạng dải", "Không đều"],
                "Size": ["Đường kính 0.5 cm"],
                "Boundary": ["Rõ", "Không rõ", "Đều", "Không đều"],
                "Quantity": ["Đơn độc", "Vài tổn thương", "Nhiều tổn thương"],
                "Distribution": ["Khu trú", "Rải rác", "Dạng cụm", "Dạng đường"],
            },
        },
        "annotations": {"Category": ["Sẩn"]},
        "diagnosis_label": "Bệnh vảy nến",
        "diagnosis_candidates": ["Bệnh vảy nến", "Bệnh bạch biến", "Bệnh trứng cá", "Bệnh ghẻ"],
        "max_qa": 1,
        "used_questions": [],
        "has_marked_region": False,
    }


class PromptCompositionTests(unittest.TestCase):
    def test_catalog_covers_twenty_combinations_per_profile(self):
        expected_tasks = {
            "Lesion_Recognition", "Attribute_Recognition", "Location",
            "Lesion_Reasoning", "Diagnosis",
        }
        expected_types = {"Short_answer", "Multi_choice", "Judgement", "Fill_in_blank"}
        for profile in ("paper", "runtime"):
            items = api().catalog(profile)
            self.assertEqual(len(items), 20)
            self.assertEqual({(i["category"], i["type"]) for i in items},
                             {(task, kind) for task in expected_tasks for kind in expected_types})
            self.assertEqual(len({i["prompt_id"] for i in items}), 20)
            self.assertTrue(all(i["prompt"].strip() for i in items))

    def test_bound_context_preserves_unicode_and_literal_braces(self):
        payload = context()
        payload["annotations"]["source_note"] = "{Không sửa dữ liệu nguồn}"
        text = api().build_prompt("Lesion_Recognition", "Short_answer", context=payload)
        bound = json.loads(text.split("\nINPUT_JSON\n", 1)[1])
        self.assertEqual(bound["annotations"]["source_note"], "{Không sửa dữ liệu nguồn}")
        self.assertEqual(bound["taxonomy"]["fields"]["Category"][0], "Sẩn")

    def test_non_diagnosis_prompt_does_not_receive_disease_gold(self):
        text = api().build_prompt("Lesion_Recognition", "Short_answer", context=context())
        bound = json.loads(text.split("\nINPUT_JSON\n", 1)[1])
        self.assertNotIn("diagnosis_label", bound)
        self.assertNotIn("diagnosis_candidates", bound)

    def test_diagnosis_prompt_receives_exact_source_label(self):
        text = api().build_prompt("Diagnosis", "Short_answer", context=context())
        bound = json.loads(text.split("\nINPUT_JSON\n", 1)[1])
        self.assertEqual(bound["diagnosis_label"], "Bệnh vảy nến")

    def test_diagnosis_without_gold_is_rejected_before_generation(self):
        payload = context()
        del payload["diagnosis_label"]
        with self.assertRaises(ValueError):
            api().build_prompt("Diagnosis", "Short_answer", context=payload)

    def test_attribute_request_requires_one_of_six_fields(self):
        for field in ("Size", "Color", "Boundary", "Shape", "Quantity", "Distribution"):
            text = api().build_prompt("Attribute_Recognition", "Short_answer",
                                      context=context(), attribute_field=field)
            bound = json.loads(text.split("\nINPUT_JSON\n", 1)[1])
            self.assertEqual(bound["attribute_field"], field)
        with self.assertRaises(ValueError):
            api().build_prompt("Attribute_Recognition", "Short_answer", context=context())
        with self.assertRaises(ValueError):
            api().build_prompt("Attribute_Recognition", "Short_answer",
                               context=context(), attribute_field="Diagnosis")

    def test_rejects_unknown_task_type_profile_and_empty_dictionary(self):
        for arguments in (("Other", "Short_answer", "runtime"),
                          ("Location", "Multiple Choice", "runtime"),
                          ("Location", "Short_answer", "unknown")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                api().build_prompt(arguments[0], arguments[1], profile=arguments[2])
        payload = context()
        payload["taxonomy"]["fields"]["Location"] = []
        with self.assertRaises(ValueError):
            api().build_prompt("Location", "Short_answer", context=payload)

    def test_max_qa_is_positive_integer_not_boolean(self):
        for limit in (0, -1, True, "1"):
            payload = context()
            payload["max_qa"] = limit
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                api().build_prompt("Location", "Short_answer", context=payload)


class CatalogCLITests(unittest.TestCase):
    def test_module_exports_forty_variants_without_network(self):
        run = subprocess.run([sys.executable, "-m", "Phase_2.qa_prompts", "catalog",
                              "--profile", "all"], cwd=ROOT, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr.decode("utf-8", errors="replace"))
        items = json.loads(run.stdout.decode("utf-8"))
        self.assertEqual(len(items), 40)
        self.assertEqual(len({i["prompt_id"] for i in items}), 40)

    def test_cli_refuses_to_overwrite_existing_output(self):
        api()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "prompts.json"
            target.write_text("valuable existing data", encoding="utf-8")
            run = subprocess.run([sys.executable, "-m", "Phase_2.qa_prompts", "catalog",
                                  "--profile", "all", "--output", str(target)],
                                 cwd=ROOT, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(target.read_text(encoding="utf-8"), "valuable existing data")


if __name__ == "__main__":
    unittest.main()
