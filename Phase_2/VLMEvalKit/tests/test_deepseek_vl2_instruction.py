import ast
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "vlmeval" / "vlm" / "deepseek_vl2.py"


def load_functions(*names):
    """Load the pure prompt helper without importing the model dependencies."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), MODULE_PATH, "exec"),
        namespace,
    )
    return tuple(namespace[name] for name in names)


class DeepSeekVL2InstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_instruction, = load_functions("get_custom_instruction")
        cls.get_instruction = staticmethod(get_instruction)

    def test_vietnamese_judgement_statement_requires_yes_or_no(self):
        question = (
            "Dựa trên màu Đỏ hồng và Trắng; hình thái Tròn; nhận định tổn "
            "thương chính thuộc nhóm Sẩn viêm và Mụn mủ là phù hợp."
        )

        instruction = self.get_instruction(question, "DermNet_Test_VI")

        self.assertIn("'Có' hoặc 'Không'", instruction)

    def test_vietnamese_open_question_keeps_short_answer_instruction(self):
        instruction = self.get_instruction(
            "Những dấu hiệu hình thái nào ủng hộ loại tổn thương Sẩn?",
            "DermNet_Test_VI",
        )

        self.assertIn("không quá 2 câu", instruction)

    def test_english_judgement_question_requires_yes_or_no(self):
        instruction = self.get_instruction("Is this lesion malignant?", "DermNet_Test_EN")

        self.assertIn("'Yes' or 'No'", instruction)

    def test_preformatted_dermnet_prompt_is_not_given_a_second_instruction(self):
        instruction = self.get_instruction(
            "Is this lesion malignant?\n[DermNet answer format] "
            "Answer with exactly 'Yes' or 'No'.",
            "DermNet_Test_EN",
        )

        self.assertEqual("", instruction)

    def test_non_dermnet_dataset_is_not_given_medical_answer_instructions(self):
        self.assertEqual("", self.get_instruction("What is shown?", "MMMU_DEV_VAL"))

    def test_medical_system_prompt_does_not_override_closed_answer_format(self):
        get_system_prompt, = load_functions("get_system_prompt")

        prompt = get_system_prompt("DermNet_Test_VI")

        self.assertIn("answer-format instruction", prompt)
        self.assertNotIn("complete, grammatically correct", prompt)
        self.assertEqual("", get_system_prompt("MMMU_DEV_VAL"))


if __name__ == "__main__":
    unittest.main()
