import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class DeepSeekVLLMTest(unittest.TestCase):
    def test_real_vllm_call_preserves_question_and_image(self):
        file = Path(__file__).parents[1] / 'vlmeval/vlm/deepseek_vllm_backend.py'
        spec = importlib.util.spec_from_file_location('backend', file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = MagicMock()
        engine.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text=' Có ')])]
        factory = MagicMock(return_value=engine)
        sampling = MagicMock()
        with patch.dict(sys.modules, {'vllm': SimpleNamespace(LLM=factory, SamplingParams=sampling)}):
            backend = module.DeepSeekVLLMBackend('deepseek-ai/deepseek-vl2-tiny')
        self.assertEqual('bfloat16', factory.call_args.kwargs['dtype'])
        self.assertNotIn('quantization', factory.call_args.kwargs)
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'test.png'
            Image.new('RGB', (8, 8)).save(image)
            question = '[DermNet answer format] Chỉ trả lời Có hoặc Không.'
            result = backend.generate([{'type': 'image', 'value': str(image)}, {'type': 'text', 'value': question}])
            self.assertEqual('Có', result)
            request = engine.generate.call_args.args[0][0]
            self.assertIn(question, request['prompt'])
            self.assertEqual((8, 8), request['multi_modal_data']['image'].size)
        with self.assertRaises(ValueError):
            backend.generate([{'type': 'text', 'value': 'missing image'}])
