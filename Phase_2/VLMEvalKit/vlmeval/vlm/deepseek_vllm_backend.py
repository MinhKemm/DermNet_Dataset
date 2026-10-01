"""Single-image DeepSeek-VL2 backend following vLLM's official example."""
import os

from PIL import Image


class DeepSeekVLLMBackend:
    def __init__(self, model_path, max_tokens=512):
        from vllm import LLM, SamplingParams

        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        self.engine = LLM(
            model=model_path,
            trust_remote_code=True,
            hf_overrides={'architectures': ['DeepseekVLV2ForCausalLM']},
            dtype='bfloat16',
            tensor_parallel_size=1,
            max_model_len=4096,
            max_num_seqs=1,
            limit_mm_per_prompt={'image': 1},
            gpu_memory_utilization=float(os.environ.get('DERMNET_VLLM_GPU_UTIL', '0.80')),
        )
        self.sampling = SamplingParams(temperature=0, max_tokens=max_tokens, seed=0)

    def generate(self, message):
        if any(item.get('type') not in ('image', 'text') for item in message):
            raise ValueError('DeepSeek vLLM backend supports single-turn image/text only')
        paths = [item['value'] for item in message if item['type'] == 'image']
        if len(paths) != 1:
            raise ValueError('DeepSeek vLLM backend requires exactly one image')
        question = '\n'.join(item['value'] for item in message if item['type'] == 'text')
        if not question.strip():
            raise ValueError('Question is empty')
        with Image.open(paths[0]) as source:
            picture = source.convert('RGB')
        try:
            outputs = self.engine.generate(
                [{'prompt': f'<|User|>: <image>\n{question}\n\n<|Assistant|>:',
                  'multi_modal_data': {'image': picture}}],
                sampling_params=self.sampling,
                use_tqdm=False,
            )
            answer = outputs[0].outputs[0].text.strip()
            if not answer:
                raise ValueError('vLLM returned an empty prediction')
            return answer
        finally:
            picture.close()
