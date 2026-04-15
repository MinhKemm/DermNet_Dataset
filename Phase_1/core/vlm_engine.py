"""
VLM Engine — multi-provider VLM cho Phase 1

Providers:
  anthropic  — Claude Opus / Sonnet
              (base_url tùy chọn: Groq, OpenRouter, Together…)
  openai     — GPT-4o / GPT-4o-mini

Đọc token từ .env.
"""

import os
import json
import re
import time
import base64
import requests
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────
#  Token & endpoint từ .env
# ──────────────────────────────────────────────
ANTHROPIC_TOKEN  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "").rstrip("/")

OPENAI_TOKEN = os.getenv("OPENAI_API_KEY", "")


# ──────────────────────────────────────────────
#  Image encoder helper
# ──────────────────────────────────────────────
def encode_image(image_path: str, max_size=(1024, 1024)) -> str:
    """Resize + encode ảnh sang base64 JPEG."""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img.thumbnail(max_size, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ──────────────────────────────────────────────
#  VLMEngine
# ──────────────────────────────────────────────
class VLMEngine:
    def __init__(self):
        self.model_id   = None
        self.provider   = None
        self._clients   = {}   # lazy clients

    def flush_memory(self):
        self._clients.clear()
        print(f"--- [VLMEngine] flushed ({self.provider}) ---")

    # ──────────────────────────────────────────
    #  load_model(provider, model_id)
    # ──────────────────────────────────────────
    def load_model(self, provider: str, model_id: str):
        self.provider = provider
        self.model_id = model_id

        if provider == "anthropic":
            self._load_anthropic()

        elif provider == "openai":
            self._load_openai()

        else:
            raise ValueError(
                f"[VLMEngine] Provider '{provider}' không hỗ trợ. "
                "Dùng: 'anthropic' | 'openai'"
            )

    def _load_anthropic(self):
        """Khởi tạo Anthropic client — hỗ trợ base_url tùy chỉnh."""
        if not ANTHROPIC_TOKEN:
            raise RuntimeError(
                "[VLMEngine] ANTHROPIC_API_KEY chưa đặt trong .env"
            )

        try:
            from anthropic import Anthropic
        except ImportError:
            raise RuntimeError(
                "[VLMEngine] pip install anthropic"
            )

        kwargs = {"api_key": ANTHROPIC_TOKEN}

        # Dùng base_url tùy chỉnh nếu có (Groq, OpenRouter, Together…)
        if ANTHROPIC_BASE_URL:
            kwargs["base_url"] = ANTHROPIC_BASE_URL
            backend = ANTHROPIC_BASE_URL.split("//")[1].split("/")[0]
            print(f"[VLMEngine] ✅ Anthropic — model: {self.model_id} | base_url: {backend}")
        else:
            print(f"[VLMEngine] ✅ Anthropic — model: {self.model_id} | endpoint: api.anthropic.com")

        self._clients["anthropic"] = Anthropic(**kwargs)

    def _load_openai(self):
        """Khởi tạo OpenAI client."""
        if not OPENAI_TOKEN:
            raise RuntimeError(
                "[VLMEngine] OPENAI_API_KEY chưa đặt trong .env"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("[VLMEngine] pip install openai")

        self._clients["openai"] = OpenAI(api_key=OPENAI_TOKEN)
        print(f"[VLMEngine] ✅ OpenAI — model: {self.model_id}")

    # ──────────────────────────────────────────
    #  call_vlm(system_prompt, user_prompt, image_path=None)
    # ──────────────────────────────────────────
    def call_vlm(self, system_prompt: str, user_prompt: str,
                 image_path: str = None) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt, image_path)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, image_path)
        return f"LỖI: Provider '{self.provider}' không hỗ trợ."

    # ── Anthropic (Claude) ─────────────────────
    def _call_anthropic(self, system_prompt, user_prompt, image_path):
        cli = self._clients.get("anthropic")
        if cli is None:
            return "LỖI: Anthropic client chưa khởi tạo."

        content = [{"type": "text", "text": user_prompt}]
        if image_path:
            content.append({
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       encode_image(image_path),
                }
            })

        try:
            resp = cli.messages.create(
                model       = self.model_id,
                max_tokens  = 4096,
                temperature = 0.1,
                system      = system_prompt,
                messages     = [{"role": "user", "content": content}],
            )

            # Lặp qua các khối nội dung để tìm khối "text" thực sự
            final_text = ""
            for block in resp.content:
                if hasattr(block, 'text'):
                    final_text += block.text
            
            return final_text.strip() if final_text else "LỖI: Không tìm thấy nội dung văn bản."

            # return resp.content[0].text.strip()
        except Exception as e:
            return f"LỖI Anthropic: {e}"

    # ── OpenAI (GPT-4o) ────────────────────────
    def _call_openai(self, system_prompt, user_prompt, image_path):
        cli = self._clients.get("openai")
        if cli is None:
            return "LỖI: OpenAI client chưa khởi tạo."

        content = [{"type": "text", "text": user_prompt}]
        if image_path:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                }
            })

        try:
            resp = cli.chat.completions.create(
                model       = self.model_id,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": content},
                ],
                temperature = 0.1,
                max_tokens = 4096,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"LỖI OpenAI: {e}"

    # ──────────────────────────────────────────
    #  extract_json(text) → dict
    # ──────────────────────────────────────────
    def extract_json(self, text: str) -> dict:
        """Trích JSON từ text, chấp nhận markdown code block hoặc thuần text."""
        try:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            json_str = match.group(1).strip() if match else None

            if not json_str:
                start = text.find('{')
                end   = text.rfind('}')
                json_str = text[start:end + 1] if start != -1 else text

            json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
            parsed   = json.loads(json_str)

            if "JSON_EXTRACTION" not in parsed:
                parsed = {"JSON_EXTRACTION": parsed}
            return parsed

        except Exception as e:
            print(f"[-] LỖI PARSE JSON: {e}")
            return {"error": f"JSON parse failed: {e}", "raw": text[:300]}

    # ──────────────────────────────────────────
    #  debug_log(phase, raw_text, ...)
    # ──────────────────────────────────────────
    def debug_log(self, phase, raw_text,
                  image_name="", disease_name=""):
        """Lưu response thô ra file để inspect."""
        debug_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "debug_outputs"
        ))
        os.makedirs(debug_dir, exist_ok=True)

        safe = lambda s: re.sub(r'[^\w\-_.]', '_', s or "")
        ts   = time.strftime("%Y%m%d_%H%M%S")
        fname = f"DEBUG_P{phase}_{safe(disease_name)}__{safe(image_name)}__{ts}.txt"
        path  = os.path.join(debug_dir, fname)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"PROVIDER: {self.provider}\n"
                    f"MODEL   : {self.model_id}\n"
                    f"PHASE   : {phase}\n"
                    f"IMAGE   : {image_name}\n"
                    f"DISEASE : {disease_name}\n"
                    f"TIME    : {ts}\n"
                    f"{'='*60}\n"
                    f"{raw_text}")

        print(f"[DEBUG P{phase}] Đã lưu → {path}")
        return path
