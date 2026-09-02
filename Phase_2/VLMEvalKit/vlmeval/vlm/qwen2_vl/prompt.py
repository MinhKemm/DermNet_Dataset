from __future__ import annotations


class Qwen2VLPromptMixin:
    """
    Mixin class for Qwen2VLChat to build custom prompt for different datasets.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            if dataset is not None and 'LEGO' in dataset:
                return False
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:  # MME has it's own prompt
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet', 'ChartQAPro', 'ChartQAPro_CoT', 'ChartQAPro_PoT', 'ChartMuseum'}:  # noqa: E501
            return True
        return True 

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""

        import string

        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        # --- THAY ĐỔI PROMPT Ở ĐÂY ---
        MCQ_CN_PROMPT = '\n[QUAN TRỌNG]: CHỈ IN RA ĐÁP ÁN CHÍNH XÁC. TUYỆT ĐỐI KHÔNG GIẢI THÍCH.'
        MCQ_EN_PROMPT = '\n[IMPORTANT]: ONLY OUTPUT THE CORRECT ANSWER. DO NOT EXPLAIN.'

        import string

        import pandas as pd

        def cn_string(s):
            import re

            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Ép trả lời Đúng/Sai bằng Tiếng Việt"""
        YORN_PROMPT = "\n\n[QUY TẮC BẮT BUỘC]: Trả lời bằng TIẾNG VIỆT. Nếu là nhận định đúng/sai, CHỈ in ra duy nhất 1 từ 'Đúng' hoặc 'Sai'. TUYỆT ĐỐI KHÔNG giải thích, không viết tiếng Anh."

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question + YORN_PROMPT))
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Ép trả lời tên bệnh/tổn thương ngắn gọn bằng Tiếng Việt"""
        VQA_PROMPT = "\n\n[QUY TẮC BẮT BUỘC]: Trả lời bằng TIẾNG VIỆT. CHỈ IN RA DUY NHẤT TÊN TỔN THƯƠNG / BỆNH HOẶC CỤM TỪ CẦN ĐIỀN (từ 1 đến 5 từ). TUYỆT ĐỐI KHÔNG giải thích, không dịch hay viết tiếng Anh."

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question + VQA_PROMPT))
        return msgs