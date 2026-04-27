import base64
import os
from pathlib import Path

import requests

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

API_KEY = os.getenv("OPENAI_TOKEN") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://apikey.click/v1").rstrip("/")
MODEL = os.getenv("MODEL", "gpt-5.5")

if not API_KEY:
    raise ValueError("Missing OPENAI_TOKEN or OPENAI_API_KEY in .env")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    else:
        return "image/jpeg"


def test_image(img_path: str):
    image_file = Path(img_path)

    if not image_file.is_file():
        print(f"❌ IMAGE FAIL: File not found: {image_file}")
        return

    img_b64 = encode_image(str(image_file))
    mime_type = get_mime_type(str(image_file))

    url = f"{BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300,
    }

    try:
        res = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        print("STATUS:", res.status_code)
        print("RAW:", res.text)

        if not res.ok:
            print("❌ IMAGE FAIL")
            return

        data = res.json()
        answer = data["choices"][0]["message"]["content"]

        print("\n✅ IMAGE OK:")
        print(answer)

    except Exception as e:
        print("❌ IMAGE FAIL:")
        print(repr(e))


if __name__ == "__main__":
    test_image("/Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/images/Acanthoma fissuratum/acanthoma-fissuratum-01.jpg")