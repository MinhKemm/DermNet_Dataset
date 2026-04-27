"""
VLM Engine — multi-provider VLM cho Phase 1

Providers:
  openai     — GPT-4o / GPT-4o-mini (Qua API trung gian)

Đọc token từ .env và sử dụng requests thuần để tránh bị Proxy chặn.
"""

import os
import json
import re
import time
import base64
import requests
from PIL import Image
import io
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: str = ".env"):
        env_file = Path(dotenv_path)
        if not env_file.is_file():
            return

        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

load_dotenv()

OPENAI_TOKEN = os.getenv("OPENAI_TOKEN") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://apikey.click/v1").rstrip("/")
MODEL = os.getenv("MODEL", "gpt-5.5")

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

    def flush_memory(self):
        print(f"--- [VLMEngine] flushed ({self.provider}) ---")

    # ──────────────────────────────────────────
    #  load_model(provider, model_id)
    # ──────────────────────────────────────────
    def load_model(self, provider: str, model_id: str):
        self.provider = provider
        self.model_id = model_id

        if provider == "openai":
            self._load_openai()
        else:
            raise ValueError("[VLMEngine] Luồng mới chỉ hỗ trợ 'openai'.")

    def _load_openai(self):
        if not OPENAI_TOKEN:
            raise RuntimeError("[VLMEngine] OPENAI_API_KEY chưa đặt trong .env")
        
        print(f"[VLMEngine] ✅ Khởi tạo HTTP Requests — model: {self.model_id} | base_url: {OPENAI_BASE_URL}")

    def call_vlm(self, system_prompt: str, user_prompt: str, image_path: str = None) -> str:
        if self.provider != "openai":
            return "LỖI: Cấu hình provider không hợp lệ."

        url = f"{OPENAI_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_TOKEN}",
            "Content-Type": "application/json"
        }

        content = [{"type": "text", "text": user_prompt}]
        
        if image_path:
            # Tự động xác định MIME type dựa trên đuôi file
            ext = Path(image_path).suffix.lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encode_image(image_path)}"
                }
            })

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }

        try:
            # Sử dụng requests thay vì OpenAI SDK
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if not resp.ok:
                return f"LỖI OpenAI (HTTP {resp.status_code}): {resp.text}"
                
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            return f"LỖI OpenAI (Requests): {e}"

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