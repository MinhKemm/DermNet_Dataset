"""Structural safeguards: these tests do not assert clinical image accuracy."""

import copy
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Phase_2.tests.test_qa_prompts import ROOT, api, context


def response(category="Lesion_Recognition", kind="Short_answer"):
    field, label = {
        "Lesion_Recognition": ("Category", "Sẩn"),
        "Attribute_Recognition": ("Color", "Màu đỏ"),
        "Location": ("Location", "Mu bàn tay"),
        "Lesion_Reasoning": ("Category", "Sẩn"),
        "Diagnosis": ("Diagnosis", "Bệnh vảy nến"),
    }[category]
    text = "Tổn thương nhô cao trong ảnh." if category == "Lesion_Reasoning" else label
    qa = {
        "category": category, "type": kind, "sub_category": field,
        "question": "Đặc điểm của tổn thương trong ảnh là gì?",
        "options": {}, "answer": text, "answer_label": text,
        "source_labels": [label], "target": "tổn thương giữa ảnh",
        "scope": "whole_image", "evidence": ["Quan sát được tổn thương nhô cao."],
        "rationale": "Dấu hiệu gồ quan sát được hỗ trợ phân biệt với dát.",
        "status": "candidate",
    }
    if kind == "Multi_choice":
        choices = {
            "Lesion_Recognition": ["Dát", "Mảng", "Vảy da"],
            "Attribute_Recognition": ["Màu trắng", "Màu đen", "Màu tím"],
            "Location": ["Lòng bàn tay", "Má", "Trán"],
            "Lesion_Reasoning": ["Tổn thương chỉ đổi màu.", "Không thấy độ gồ.", "Bề mặt hoàn toàn phẳng."],
            "Diagnosis": ["Bệnh bạch biến", "Bệnh trứng cá", "Bệnh ghẻ"],
        }[category]
        qa["options"] = dict(zip("ABCD", [text] + choices))
        qa["answer"] = "A"
    elif kind == "Judgement":
        qa["answer"] = qa["answer_label"] = "Có"
        if category == "Diagnosis":
            qa["claim_label"] = label
    elif kind == "Fill_in_blank":
        qa["question"] = "Đặc điểm được hỏi là ____."
    return {
        "schema_version": "2.0.0", "image_id": "example-001",
        "taxonomy_version": "fixture-reviewed-1", "qas": [qa],
        "skipped": [], "needs_review": [],
    }


class ResponseValidationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(api(), "validate_response", None)),
                        "The response validator is not implemented")

    def errors(self, payload, category="Lesion_Recognition", kind="Short_answer", ctx=None):
        return api().validate_response(payload, ctx or context(), category, kind,
                                       attribute_field="Color" if category == "Attribute_Recognition" else None)

    def test_accepts_contract_for_all_twenty_combinations(self):
        for category in ("Lesion_Recognition", "Attribute_Recognition", "Location",
                         "Lesion_Reasoning", "Diagnosis"):
            for kind in ("Short_answer", "Multi_choice", "Judgement", "Fill_in_blank"):
                with self.subTest(category=category, kind=kind):
                    self.assertEqual(self.errors(response(category, kind), category, kind), [])

    def test_rejects_wrong_image_taxonomy_or_schema_version(self):
        for field in ("image_id", "taxonomy_version", "schema_version"):
            payload = response()
            payload[field] = "wrong"
            with self.subTest(field=field):
                self.assertTrue(self.errors(payload))

    def test_rejects_changed_diagnosis_in_every_format(self):
        for kind in ("Short_answer", "Multi_choice", "Judgement", "Fill_in_blank"):
            payload = response("Diagnosis", kind)
            payload["qas"][0]["source_labels"] = ["Bệnh ghẻ"]
            with self.subTest(kind=kind):
                self.assertTrue(self.errors(payload, "Diagnosis", kind))
        payload = response("Diagnosis", "Short_answer")
        payload["qas"][0]["answer"] = payload["qas"][0]["answer_label"] = "Bệnh ghẻ"
        self.assertTrue(self.errors(payload, "Diagnosis"))

    def test_judgement_diagnosis_matches_claim_against_given_gold(self):
        payload = response("Diagnosis", "Judgement")
        payload["qas"][0]["claim_label"] = "Bệnh ghẻ"
        self.assertTrue(self.errors(payload, "Diagnosis", "Judgement"))
        payload["qas"][0]["answer"] = payload["qas"][0]["answer_label"] = "Không"
        self.assertEqual(self.errors(payload, "Diagnosis", "Judgement"), [])
        payload["qas"][0]["claim_label"] = "unapproved disease"
        self.assertTrue(self.errors(payload, "Diagnosis", "Judgement"))

    def test_rejects_malformed_or_duplicate_mcq_options_and_wrong_key(self):
        for change in ({"answer": "E"}, {"answer_label": "Mảng"},
                       {"options": {"A": "Sẩn", "B": "Dát", "C": "Mảng"}},
                       {"options": {"A": "Sẩn", "B": "sẩn", "C": "Mảng", "D": "Vảy da"}},
                       {"options": {"A": "Sẩn", "B": "Màu đỏ", "C": "Mảng", "D": "Vảy da"}}):
            payload = response(kind="Multi_choice")
            payload["qas"][0].update(change)
            with self.subTest(change=change):
                self.assertTrue(self.errors(payload, kind="Multi_choice"))

    def test_respects_assigned_answer_position(self):
        ctx = context()
        ctx["correct_option_position"] = "C"
        self.assertTrue(self.errors(response(kind="Multi_choice"), kind="Multi_choice", ctx=ctx))

    def test_fill_blank_has_exactly_one_blank(self):
        for question in ("Không có chỗ trống.", "Điền ____ và ____."):
            payload = response(kind="Fill_in_blank")
            payload["qas"][0]["question"] = question
            self.assertTrue(self.errors(payload, kind="Fill_in_blank"))

    def test_cannot_change_requested_task_type_field_or_status(self):
        for field, value in (("category", "Diagnosis"), ("type", "Multiple Choice"),
                             ("sub_category", "Shape"), ("status", "approved")):
            payload = response()
            payload["qas"][0][field] = value
            with self.subTest(field=field):
                self.assertTrue(self.errors(payload))

    def test_reasoning_requires_evidence_and_explanation_not_only_a_label(self):
        for change in ({"evidence": []}, {"rationale": ""},
                       {"answer": "Sẩn", "answer_label": "Sẩn"}):
            payload = response("Lesion_Reasoning")
            payload["qas"][0].update(change)
            self.assertTrue(self.errors(payload, "Lesion_Reasoning"))

    def test_reasoning_mcq_cannot_mix_explanations_with_lesion_name_options(self):
        payload = response("Lesion_Reasoning", "Multi_choice")
        payload["qas"][0]["options"]["B"] = "Dát"
        self.assertTrue(self.errors(payload, "Lesion_Reasoning", "Multi_choice"))

    def test_quantity_mcq_does_not_invent_fourth_label_for_three_label_dictionary(self):
        payload = response("Attribute_Recognition", "Multi_choice")
        payload["qas"][0].update(
            sub_category="Quantity", source_labels=["Đơn độc"], answer_label="Đơn độc",
            options={"A": "Đơn độc", "B": "Vài tổn thương", "C": "Nhiều tổn thương", "D": "Không có"})
        ctx = context()
        ctx["quantity_policy"] = {"reviewed": True, "visible_to_answerer": True,
                                  "description": "Phạm vi và ngưỡng đã được cấp."}
        self.assertTrue(api().validate_response(payload, ctx, "Attribute_Recognition", "Multi_choice",
                                               attribute_field="Quantity"))

    def test_rejects_unmarked_region_duplicates_or_qa_over_limit(self):
        payload = response()
        payload["qas"][0]["scope"] = "marked_region"
        self.assertTrue(self.errors(payload))
        payload = response()
        payload["qas"].append(copy.deepcopy(payload["qas"][0]))
        self.assertTrue(self.errors(payload))
        ctx = context()
        ctx["used_questions"] = [payload["qas"][0]["question"]]
        self.assertTrue(self.errors(response(), ctx=ctx))

    def test_handles_untrusted_shapes_without_crashing(self):
        for payload in (None, [], {}, {"qas": None}, response()):
            if isinstance(payload, dict) and payload.get("qas"):
                payload["qas"] = [None, {"options": []}]
            with self.subTest(payload=payload):
                self.assertTrue(self.errors(payload))

    def test_empty_qas_requires_explained_skip_or_review(self):
        payload = response()
        payload["qas"] = []
        self.assertTrue(self.errors(payload))
        payload["skipped"] = [{"reason": "image_unavailable", "detail": "Ảnh không được đính kèm."}]
        self.assertEqual(self.errors(payload), [])

    def test_blocking_skip_cannot_coexist_with_generated_qa(self):
        for reason in ("image_unavailable", "image_label_conflict"):
            payload = response()
            payload["skipped"] = [{"reason": reason}]
            with self.subTest(reason=reason):
                self.assertTrue(self.errors(payload))

    def test_size_and_quantity_need_explicit_supporting_inputs(self):
        for field, label, support in (("Size", "Đường kính 0.5 cm", "measurement"),
                                      ("Quantity", "Đơn độc", "quantity_policy")):
            ctx = context()
            payload = response("Attribute_Recognition")
            qa = payload["qas"][0]
            qa.update(sub_category=field, source_labels=[label], answer=label, answer_label=label)
            self.assertTrue(api().validate_response(payload, ctx, "Attribute_Recognition", "Short_answer",
                                                    attribute_field=field))
            ctx[support] = {"visible_to_answerer": True, "reviewed": True,
                            "description": "Căn cứ được cung cấp trong đầu vào người trả lời."}
            self.assertEqual(api().validate_response(payload, ctx, "Attribute_Recognition", "Short_answer",
                                                     attribute_field=field), [])
            ctx[support]["visible_to_answerer"] = False
            self.assertTrue(api().validate_response(payload, ctx, "Attribute_Recognition", "Short_answer",
                                                    attribute_field=field))


class TSVExportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(api(), "to_tsv", None)), "The TSV adapter is not implemented")

    def test_mcq_options_are_visible_in_question_and_key_preserved(self):
        text = api().to_tsv(response(kind="Multi_choice"), context(),
                            "Lesion_Recognition", "Multi_choice", start_index=100)
        row = list(csv.DictReader(io.StringIO(text), delimiter="\t"))[0]
        self.assertEqual(row["index"], "100")
        self.assertEqual(row["type"], "Multi_choice")
        self.assertEqual(row["answer"], "A")
        self.assertIn("A. Sẩn", row["question"])
        self.assertIn("D. Vảy da", row["question"])
        self.assertNotIn("evidence", row)
        self.assertNotIn("source_labels", row)

    def test_export_refuses_invalid_qa_missing_image_path_or_invalid_start(self):
        invalid = response()
        invalid["qas"][0]["answer"] = "U"
        with self.assertRaises(ValueError):
            api().to_tsv(invalid, context(), "Lesion_Recognition", "Short_answer")
        ctx = context()
        del ctx["image_path"]
        with self.assertRaises(ValueError):
            api().to_tsv(response(), ctx, "Lesion_Recognition", "Short_answer")
        with self.assertRaises(ValueError):
            api().to_tsv(response(), context(), "Lesion_Recognition", "Short_answer", start_index=True)

    def test_cli_validation_and_export_use_real_json_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "context.json").write_text(json.dumps(context(), ensure_ascii=False), encoding="utf-8")
            (directory / "qa.json").write_text(json.dumps(response(), ensure_ascii=False), encoding="utf-8")
            arguments = ["--task", "Lesion_Recognition", "--type", "Short_answer",
                         "--context", str(directory / "context.json"), "--response", str(directory / "qa.json")]
            for command in ("validate", "to-tsv"):
                run = subprocess.run([sys.executable, "-m", "Phase_2.qa_prompts", command] + arguments,
                                     cwd=ROOT, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stderr.decode("utf-8", errors="replace"))
            bad = response()
            bad["image_id"] = "wrong-image"
            (directory / "qa.json").write_text(json.dumps(bad), encoding="utf-8")
            run = subprocess.run([sys.executable, "-m", "Phase_2.qa_prompts", "validate"] + arguments,
                                 cwd=ROOT, capture_output=True)
            self.assertNotEqual(run.returncode, 0)


if __name__ == "__main__":
    unittest.main()
