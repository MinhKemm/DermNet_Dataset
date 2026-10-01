"""Test loader wiring with fake weights; no GPU or model downloads required."""
import ast
import logging
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

KIT = Path(__file__).parents[1]


def constructor(path, name, extra=None):
    tree = ast.parse((KIT / path).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    scope = {'torch': Mock(bfloat16='bf16'), 'logging': logging, 'warnings': Mock()}
    scope.update(extra or {})
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(path), 'exec'), scope)
    return scope['__init__']


class LegacyModelLoadingTest(unittest.TestCase):
    def test_all_manifest_models_are_registered(self):
        tree = ast.parse((KIT / 'vlmeval/config.py').read_text(encoding='utf-8'))
        keys = {key.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
        jobs = [line.split('|') for line in (KIT / 'scripts/dermnet_jobs.txt').read_text().splitlines()
                if line and not line.startswith('#')]
        self.assertEqual(set(), {job[1] for job in jobs} - keys)

    def test_janus_four_bit_is_loading_option_not_generation_option(self):
        transformers = Mock()
        models = Mock()
        bot = types.SimpleNamespace(check_install=lambda: None)
        init = constructor('vlmeval/vlm/janus.py', 'Janus',
                           {'AutoModelForCausalLM': transformers.AutoModelForCausalLM})
        with patch.dict('sys.modules', {'transformers': transformers, 'janus.models': models}):
            init(bot, 'deepseek-ai/Janus-Pro-7B', load_in_4bit=True)
        args = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertIn('quantization_config', args)
        self.assertEqual({'': 0}, args['device_map'])
        self.assertNotIn('load_in_4bit', bot.kwargs)
        transformers.AutoModelForCausalLM.from_pretrained.return_value.to.assert_not_called()
        self.assertTrue(transformers.BitsAndBytesConfig.call_args.kwargs['load_in_4bit'])

    def test_phi_four_bit_is_loading_option(self):
        transformers = Mock()
        bot = types.SimpleNamespace()
        with patch.dict('sys.modules', {'transformers': transformers}):
            constructor('vlmeval/vlm/phi3_vision.py', 'Phi3_5Vision')(
                bot, load_in_4bit=True, attn_implementation='eager')
        args = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertIn('quantization_config', args)
        self.assertEqual('eager', args['_attn_implementation'])
        self.assertNotIn('load_in_4bit', bot.kwargs)

    def test_llava_passes_four_bit_to_official_builder(self):
        builder, mm = Mock(), Mock()
        model = Mock()
        builder.load_pretrained_model.return_value = (Mock(), model, Mock(), 2048)
        init = constructor('vlmeval/vlm/llava/llava.py', 'LLaVA',
                           {'osp': Mock(), 'splitlen': lambda value: 2})
        bot = types.SimpleNamespace()
        with patch.dict('sys.modules', {'llava.mm_utils': mm, 'llava.model.builder': builder}):
            init(bot, load_in_4bit=True)
        self.assertTrue(builder.load_pretrained_model.call_args.kwargs['load_4bit'])
        self.assertEqual('cuda:0', builder.load_pretrained_model.call_args.kwargs['device_map'])
        model.cuda.assert_not_called()
        self.assertNotIn('load_in_4bit', bot.kwargs)

    def test_gemma_unified_uses_auto_model(self):
        transformers = Mock()
        bot = types.SimpleNamespace()
        with patch.dict('sys.modules', {'transformers': transformers}):
            constructor('vlmeval/vlm/gemma.py', 'Gemma4')(
                bot, model_path='google/gemma-4-12B', use_vllm=False, use_auto_model=True)
        transformers.AutoModelForMultimodalLM.from_pretrained.assert_called_once()
        transformers.Gemma4ForConditionalGeneration.from_pretrained.assert_not_called()
        self.assertNotIn('use_auto_model', bot.kwargs)


if __name__ == '__main__':
    unittest.main()
