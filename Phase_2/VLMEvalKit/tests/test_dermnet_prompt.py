import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "vlmeval"
    / "dataset"
    / "utils"
    / "dermnet_prompt.py"
)
DATASET_MODULE_PATH = Path(__file__).parents[1] / "vlmeval" / "dataset" / "dermnet.py"


def load_prompt_module():
    spec = importlib.util.spec_from_file_location("dermnet_prompt", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DermNetPromptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = load_prompt_module()

    def test_multichoice_requires_only_concatenated_letters(self):
        instruction = self.prompt.answer_instruction("Multi_choice", "vi")
        self.assertIn("A hoặc ACD", instruction)
        self.assertIn("không", instruction.lower())

    def test_judgement_uses_dataset_language(self):
        self.assertIn(
            "'Có' hoặc 'Không'", self.prompt.answer_instruction("Judgement", "vi")
        )
        self.assertIn(
            "'Yes' or 'No'", self.prompt.answer_instruction("Judgement", "en")
        )

    def test_fill_in_blank_requires_only_missing_phrase(self):
        instruction = self.prompt.answer_instruction("Fill_in_blank", "en")
        self.assertIn("missing term or phrase only", instruction)

    def test_short_answer_is_concise_but_not_forced_to_one_word(self):
        instruction = self.prompt.answer_instruction("Short_answer", "vi")
        self.assertIn("cụm từ", instruction)
        self.assertNotIn("một từ", instruction)

    def test_append_instruction_is_idempotent(self):
        question = "Tổn thương có phải là sẩn không?"
        once = self.prompt.append_answer_instruction(question, "Judgement", "vi")
        twice = self.prompt.append_answer_instruction(once, "Judgement", "vi")
        self.assertEqual(once, twice)

    def test_dataset_prompt_uses_row_type_and_dataset_language(self):
        import ast

        class StubCustomVQADataset:
            def build_prompt(self, line):
                return [
                    {"type": "image", "value": line["image_path"]},
                    {"type": "text", "value": line["question"]},
                ]

        tree = ast.parse(DATASET_MODULE_PATH.read_text(encoding="utf-8"))
        dataset_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DermNetDataset"
        )
        namespace = {
            "CustomVQADataset": StubCustomVQADataset,
            "append_answer_instruction": self.prompt.append_answer_instruction,
        }
        exec(
            compile(
                ast.Module(body=[dataset_class], type_ignores=[]),
                DATASET_MODULE_PATH,
                "exec",
            ),
            namespace,
        )
        dataset = namespace["DermNetDataset"]()
        dataset.dataset_name = "DermNet_Val_EN"

        message = dataset.build_prompt(
            {
                "image_path": "image.jpg",
                "question": "The diagnosis is consistent with psoriasis.",
                "type": "Judgement",
            }
        )

        self.assertTrue(dataset.force_use_dataset_prompt)
        self.assertIn("'Yes' or 'No'", message[-1]["value"])

    def test_llava_and_vintern_do_not_duplicate_dataset_instruction(self):
        import ast

        prompt = (
            "Tổn thương có phải là sẩn không?\n[DermNet answer format] "
            "Chỉ trả lời chính xác 'Có' hoặc 'Không'."
        )
        for relative_path, class_name in (
            ("vlmeval/vlm/llava/llava.py", "LLaVA"),
            ("vlmeval/vlm/vintern_chat.py", "VinternChat"),
        ):
            model_path = Path(__file__).parents[1] / relative_path
            tree = ast.parse(model_path.read_text(encoding="utf-8"))
            model_class = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            format_method = next(
                node
                for node in model_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "_format_prompt"
            )
            namespace = {}
            exec(
                compile(
                    ast.Module(body=[format_method], type_ignores=[]),
                    model_path,
                    "exec",
                ),
                namespace,
            )
            self.assertEqual(prompt, namespace["_format_prompt"](None, prompt))


if __name__ == "__main__":
    unittest.main()
