import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock


class HuatuoAdapterTest(unittest.TestCase):
    def test_forwards_prompt_and_image_without_history(self):
        source = Path(__file__).parents[1] / 'vlmeval/vlm/huatuo_vision.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        scope = {'BaseModel': object, 'Path': Path}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), scope)
        adapter = scope['HuatuoGPTVision'].__new__(scope['HuatuoGPTVision'])
        adapter.bot = Mock()
        adapter.bot.inference.return_value = [' Yes ']
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.jpg'
            image.touch()
            prompt = '[DermNet answer format] Answer with exactly Yes or No.'
            result = adapter.generate_inner([
                {'type': 'image', 'value': str(image)},
                {'type': 'text', 'value': prompt},
            ])
            self.assertEqual('Yes', result)
            adapter.bot.clear_history.assert_called_once()
            adapter.bot.inference.assert_called_once_with(prompt, [str(image)])


if __name__ == '__main__':
    unittest.main()
