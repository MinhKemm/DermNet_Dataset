# src/models/base_client.py
from abc import ABC, abstractmethod

class BaseVLMClient(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.model_id = config.get("model_id")
        self.temperature = config.get("temperature", 0.0)

    @abstractmethod
    def generate(self, image_path: str, system_prompt: str, few_shot_msgs: list, user_prompt: str) -> str:
        pass