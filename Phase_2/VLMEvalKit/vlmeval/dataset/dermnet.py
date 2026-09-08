from .image_vqa import CustomVQADataset
from .utils.dermnet_prompt import append_answer_instruction


class DermNetDataset(CustomVQADataset):
    """DermNet dataset with a row-type-aware answer-format instruction."""

    TYPE = "VQA"
    force_use_dataset_prompt = True
    DATASETS = (
        "DermNet_Val_4k",
        "DermNet_Val_4k-2_mac_relative",
        "DermNet_Val_4k_en",
        "DermNet_Test",
        "DermNet_Test_1of3",
        "DermNet_Test_mac_relative",
        "DermNet_Test_1of3_en",
        "DermNet_Val_4k-2_Reasoning_Fix",
        "DermNet_Test_Reasoning_Fix",
    )

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASETS)

    def build_prompt(self, line):
        messages = super().build_prompt(line)
        language = "en" if self.dataset_name.endswith("_en") else "vi"
        question_type = line.get("type", "")
        for item in messages:
            if item.get("type") == "text":
                item["value"] = append_answer_instruction(
                    item["value"], question_type, language
                )
        return messages
