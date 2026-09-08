from .image_vqa import CustomVQADataset
from .utils.dermnet_prompt import append_answer_instruction


class DermNetDataset(CustomVQADataset):
    """DermNet dataset with a row-type-aware answer-format instruction."""

    TYPE = "VQA"
    force_use_dataset_prompt = True
    DATASETS = (
        "DermNet_Val_4k",
        "DermNet_Val_VI",
        "DermNet_Val_EN",
        "DermNet_Test",
        "DermNet_Test_1of3",
        "DermNet_Test_VI",
        "DermNet_Test_EN",
        "DermNet_Val_Reasoning_VI",
        "DermNet_Test_Reasoning_VI",
        "DermNet_Val_Reasoning_EN",
        "DermNet_Test_Reasoning_EN",
    )

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASETS)

    def build_prompt(self, line):
        messages = super().build_prompt(line)
        language = "en" if self.dataset_name.lower().endswith("_en") else "vi"
        question_type = line.get("type", "")
        for item in messages:
            if item.get("type") == "text":
                item["value"] = append_answer_instruction(
                    item["value"], question_type, language
                )
        return messages
