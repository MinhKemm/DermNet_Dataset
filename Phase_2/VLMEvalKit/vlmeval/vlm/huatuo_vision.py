"""Adapter for the official FreedomIntelligence/HuatuoGPT-Vision cli.py API."""
import importlib.util
import os
from pathlib import Path

from .base import BaseModel


class HuatuoGPTVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path, **kwargs):
        source = os.environ.get("HUATUO_SOURCE_DIR", "")
        entry = Path(source).expanduser().resolve() / "cli.py"
        if not source or not entry.is_file():
            raise ValueError("Set HUATUO_SOURCE_DIR to the official HuatuoGPT-Vision checkout containing cli.py")
        spec = importlib.util.spec_from_file_location("dermnet_huatuo_cli", entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.bot = module.HuatuoChatbot(model_path, device="cuda")
        self.bot.gen_kwargs.update(do_sample=False, max_new_tokens=512)
        self.bot.gen_kwargs.pop("temperature", None)
        self.bot.gen_kwargs.update(kwargs)

    def generate_inner(self, message, dataset=None):
        text = "\n".join(item["value"] for item in message if item["type"] == "text")
        images = [item["value"] for item in message if item["type"] == "image"]
        for image in images:
            if not Path(image).is_file():
                raise FileNotFoundError(image)
        self.bot.clear_history()
        answers = self.bot.inference(text, images)
        if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str):
            raise ValueError("HuatuoGPT-Vision returned an unexpected response")
        return answers[0].strip()
