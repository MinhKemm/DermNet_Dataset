"""
gemma_comparison.py — Gemma 3 27B IT tạo Gold Standard JSON

Luồng duy nhất (2 JSON):
  Input:
    • Ảnh bệnh nhân (image)
    • Disease Knowledge (file .txt)
    • Claude JSON  (Phase 2 output)
    • GPT-4o JSON (bạn chạy riêng)

  Gemma làm nhiệm vụ:
    NHÌN ẢNH + ĐỌC KIẾN THỨC + ĐỌC 2 JSON
    → SO SÁNH từng trường (Lesion_Type, Colour, Shape, Distribution, Characteristics)
    → NHẶT những giá trị ĐÚNG NHẤT từ CẢ 2 JSON
    → TẠO Gold Standard JSON — kết hợp những gì tốt nhất
    → ĐÁNH CHỈ SỐ: image_id, json_id đã có trong metadata

Đầu ra lưu tại:
  output/<Disease>/<img>_BEFORE_MERGE/      ← Claude + GPT-4o trước merge
  output/<Disease>/<img>_after_merge/       ← Gold Standard sau merge
  output/<Disease>/<img>_final.json         ← Final (= Gold Standard)
"""

import os
import sys
import json
import re
import time
import base64
import hashlib
import requests
from PIL import Image
import io

# ──────────────────────────────────────────────
#  Module path
# ──────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────
#  Config từ .env
# ──────────────────────────────────────────────
GOOGLE_AI_STUDIO_KEY = os.environ.get("GOOGLE_AI_STUDIO_KEY", "")
GEMMA4_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/openai/chat/completions"
)
GEMMA4_MODEL_ID = "gemma-3-27b-it"


# ═══════════════════════════════════════════════════════════════
#  PROMPTS — Gold Standard Merge (2 JSON)
# ═══════════════════════════════════════════════════════════════
GOLD_STANDARD_SYSTEM = (
    "Bạn là chuyên gia da liễu hình thái học cấp cao.\n\n"
    "Nhiệm vụ: TỪ 2 JSON mô tả cùng bệnh, tạo GOLD STANDARD JSON.\n\n"
    "Sử dụng:\n"
    "  (1) KIẾN THỨC BỆNH LÝ — hiểu đặc điểm đặc trưng\n"
    "  (2) HÌNH ẢNH THỰC TẾ — xác nhận giá trị nào ĐÚNG\n"
    "  (3) 2 JSON (Claude & GPT-4o) — nhặt giá trị TỐT NHẤT\n\n"
    "QUY TẮC CỐT LÕI:\n"
    "1. Đọc KIẾN THỨC BỆNH trước.\n"
    "2. QUAN SÁT ẢNH THỰC TẾ.\n"
    "3. Với MỖI TRƯỜNG, chọn giá trị TỐT NHẤT từ JSON A hoặc JSON B:\n"
    "   - Lesion_Type: chọn loại tổn thương ĐÚNG + ĐẦY ĐỦ nhất\n"
    "   - Colour: chọn màu KHỚP ẢNH NHẤT\n"
    "   - Shape: chọn hình dạng ĐÚNG NHẤT\n"
    "   - Distribution: chọn vị trí/cách sắp xếp CHÍNH XÁC NHẤT\n"
    "   - Characteristics: chọn đặc điểm PHÙ HỢP ẢNH NHẤT\n"
    "4. LOẠI BỎ giá trị suy luận, không thấy trong ảnh.\n"
    "5. GỘP thông minh: nếu A có giá trị X, B có giá trị Y (cùng đúng) → giữ CẢ 2.\n"
    "6. MỖI GIÁ TRỊ = 1-3 TỪ. Loại bỏ tính từ diễn giải.\n"
    "7. Đầu ra: DUY NHẤT một khối JSON. Không chào hỏi.\n"
    "8. Thêm _metadata: source_claude (boolean), source_gpt4o (boolean).\n"
)

GOLD_STANDARD_USER = (
    "═══════════════════════════════════════════════════════════\n"
    "  BỆNH: {disease_name}\n"
    "═══════════════════════════════════════════════════════════\n\n"
    "[IMAGE]\n\n"
    "─── KIẾN THỨC BỆNH ───────────────────────────────────\n"
    "{disease_knowledge}\n\n"
    "───────────────────────────────────────────────────────────\n"
    "─── JSON A: Claude Opus 4.6 ──────────────────────────────\n"
    "{json_a}\n\n"
    "───────────────────────────────────────────────────────────\n"
    "─── JSON B: GPT-4o ──────────────────────────────────────\n"
    "{json_b}\n\n"
    "═══════════════════════════════════════════════════════════\n"
    "Hãy NHÌN ẢNH → ĐỌC KIẾN THỨC → NHẶT GIÁ TRỊ TỐT NHẤT\n"
    "từ A và B → tạo GOLD STANDARD JSON.\n"
    "Trả về JSON (không prefix text):\n"
)

GOLD_STANDARD_FEW_SHOT = """
--- VÍ DỤ MINH HỌA ---

BỆNH: Nấm da ẩn danh

KIẾN THỨC BỆNH:
  Nấm da ẩn danh (Tinea Incognito) do dùng steroid che giấu hình thái.
  Đặc điểm: hồng ban bất quy tắc, ranh giới rõ, mụn mủ nang lông,
  teo da. Vị trí: mặt, cổ, thân mình.

JSON A (Claude):
  {"JSON_EXTRACTION":{
    "Category":"Nấm da ẩn danh",
    "Lesion_Type":["Mảng hồng ban"],
    "Colour":["Đỏ nhạt"],
    "Shape":["Bất quy tắc","Ranh giới rõ"],
    "Distribution":["Mu bàn tay","Khu trú"],
    "Characteristics":["Mụn mủ nang lông","Teo da"]
  }}

JSON B (GPT-4o):
  {"JSON_EXTRACTION":{
    "Category":"Nấm da ẩn danh",
    "Lesion_Type":["Mảng đỏ","Mụn nước"],
    "Colour":["Đỏ hồng"],
    "Shape":["Hình tròn","Bờ rõ"],
    "Distribution":["Mặt","Lan tỏa"],
    "Characteristics":["Ngứa"]
  }}

SO SÁNH VỚI ẢNH THỰC:
  ẢNH: Mảng hồng ban bất quy tắc, màu đỏ nhạt, ranh giới rõ,
        vùng mu bàn tay, có dấu teo da nhẹ.

  Lesion_Type:
    A: "Mảng hồng ban"    ✓ đúng ảnh
    B: "Mảng đỏ"         ✗ ảnh đỏ nhạt, không phải đỏ
    → GIỮ A

  Colour:
    A: "Đỏ nhạt"         ✓ khớp ảnh
    B: "Đỏ hồng"         ≈ gần đúng nhưng A chính xác hơn
    → GIỮ A

  Shape:
    A: "Bất quy tắc, Ranh giới rõ"  ✓ khớp ảnh
    B: "Hình tròn, Bờ rõ"            ✗ ảnh không tròn
    → GIỮ A

  Distribution:
    A: "Mu bàn tay, Khu trú"  ✓ khớp ảnh
    B: "Mặt, Lan tỏa"         ✗ sai vị trí
    → GIỮ A

  Characteristics:
    A: "Mụn mủ nang lông, Teo da"    ✓ thấy trong ảnh
    B: "Ngứa"                        ✗ không thấy trong ảnh (suy luận)
    → GIỮ A

GOLD STANDARD (merge A + phần tốt của B nếu có):
  → Tất cả giá trị A đều khớp ảnh, B không có giá trị nào vượt trội.
  → Giữ nguyên A.

Output:
```json
{{"JSON_EXTRACTION":{{
  "Category":"Nấm da ẩn danh",
  "Lesion_Type":["Mảng hồng ban"],
  "Colour":["Đỏ nhạt"],
  "Shape":["Bất quy tắc","Ranh giới rõ"],
  "Distribution":["Mu bàn tay","Khu trú"],
  "Characteristics":["Mụn mủ nang lông","Teo da"],
  "_metadata":{{
    "source_claude": true,
    "source_gpt4o": false
  }}
}}}}
```

--> BÂY GIỜ LÀ LƯỢT CỦA BẠN — TẠO GOLD STANDARD CHO: {disease_name}:
"""


# ═══════════════════════════════════════════════════════════════
#  GemmaGoldStandardEngine
# ═══════════════════════════════════════════════════════════════
class GemmaGoldStandardEngine:
    """
    Gemma 3 27B IT tạo Gold Standard JSON.

    Luồng:
      1. NHÌN ẢNH + ĐỌC KIẾN THỨC + ĐỌC 2 JSON (Claude + GPT-4o)
      2. SO SÁNH từng trường
      3. NHẶT giá trị ĐÚNG NHẤT từ CẢ 2 JSON
      4. TRẢ VỀ Gold Standard JSON (image_id + json_id đã có sẵn trong metadata)
    """

    def __init__(self, model_id: str = GEMMA4_MODEL_ID):
        self.model_id = model_id
        self.api_key  = GOOGLE_AI_STUDIO_KEY
        self._session = requests.Session()

    def flush_memory(self):
        print("[GemmaGold] flush_memory")

    def load_model(self):
        print(f"[GemmaGold] Model: {self.model_id}")
        if not self.api_key:
            print("[GemmaGold] ⚠️ GOOGLE_AI_STUDIO_KEY chưa đặt!")

    # ──────────────────────────────────────────────────────────
    #  create_gold_standard — luồng merge chính
    # ──────────────────────────────────────────────────────────
    def create_gold_standard(
        self,
        image_path: str,
        json_a: dict,
        json_b: dict,
        disease_name: str,
        disease_knowledge: str,
        image_id: str,
    ) -> dict:
        """
        Trả về Gold Standard JSON.
        image_id đã được sinh trước ở Phase 2 — chỉ nhận và gắn vào metadata.

        Returns:
          {
            "JSON_EXTRACTION": {
              ...fields...,
              "_metadata": {
                "image_id": "IMG_...",
                "json_id":  "JSON_IMG_..._gold_...",
                "source_claude": true,
                "source_gpt4o":  true,
                "timestamp": "...",
              }
            }
          }
        """
        self._check_api_key()
        self._check_image(image_path)

        print(f"[GemmaGold] Merge Claude + GPT-4o → Gold Standard | Bệnh: {disease_name}")
        print(f"             image_id: {image_id}")

        b64_img = encode_image(image_path)

        user_prompt = (
            GOLD_STANDARD_USER.format(
                disease_name       = disease_name,
                disease_knowledge = (disease_knowledge or "(Không có kiến thức bệnh)")[:2000],
                json_a             = json.dumps(json_a, ensure_ascii=False, indent=2),
                json_b             = json.dumps(json_b, ensure_ascii=False, indent=2),
            )
            + GOLD_STANDARD_FEW_SHOT.format(disease_name=disease_name)
        )

        messages = [
            {"role": "system", "content": GOLD_STANDARD_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    {"type": "text",     "text": user_prompt},
                ],
            },
        ]

        raw      = self._call_gemma(messages)
        gold_json = self._parse_and_enrich(raw, json_a, json_b, image_id, disease_name)

        print(f"[GemmaGold] ✅ Gold Standard created.")
        return gold_json

    # ──────────────────────────────────────────────────────────
    #  Alias tương thích với code cũ
    # ──────────────────────────────────────────────────────────
    def compare(
        self,
        image_path: str,
        json_a: dict,
        json_b: dict,
        disease_name: str,
        disease_knowledge: str,
        image_id: str = None,
    ) -> dict:
        """
        Alias cho create_gold_standard().
        Giữ backward compatibility với code cũ gọi .compare().

        Returns dict với keys: 'winning_json', 'winner', 'reason'
        """
        # Tạo image_id tạm nếu caller không truyền
        if image_id is None:
            image_id = generate_image_id_fallback(image_path)

        gold = self.create_gold_standard(
            image_path, json_a, json_b,
            disease_name, disease_knowledge, image_id,
        )

        return {
            "winning_json": gold,
            "winner":      "MERGED",
            "reason":      "Gemma merge Claude + GPT-4o → Gold Standard",
        }

    # ──────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────
    def _check_api_key(self):
        if not self.api_key:
            raise RuntimeError(
                "[GemmaGold] GOOGLE_AI_STUDIO_KEY chưa đặt trong .env. "
                "Lấy tại: https://aistudio.google.com/app/apikey"
            )

    def _check_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[GemmaGold] Không tìm thấy ảnh: {image_path}")

    def _call_gemma(self, messages: list) -> str:
        url     = f"{GEMMA4_BASE_URL}?key={self.api_key}"
        payload = {
            "model":       self.model_id,
            "messages":    messages,
            "temperature": 0.05,
            "max_tokens":  2048,
        }
        resp = self._session.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _parse_and_enrich(
        self,
        raw: str,
        json_a: dict,
        json_b: dict,  # noqa: F841 — lưu vào metadata để trace nguồn
        image_id: str,
        disease_name: str,
    ) -> dict:
        """
        Trích JSON từ Gemma response → canonicalize → bổ sung _metadata.
        image_id + json_id được gắn vào đây.
        json_b được giữ lại để tham chiếu nguồn trong metadata.
        """
        match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if match:
            json_str = re.sub(r'[\x00-\x1F\x7F]', '', match.group(1).strip())
            try:
                parsed = json.loads(json_str)
                if "JSON_EXTRACTION" not in parsed:
                    parsed = {"JSON_EXTRACTION": parsed}
            except Exception:
                parsed = _fallback_to_a(json_a)
        else:
            try:
                start = raw.find('{')
                end   = raw.rfind('}')
                if start != -1:
                    parsed = json.loads(raw[start:end+1])
                    if "JSON_EXTRACTION" not in parsed:
                        parsed = {"JSON_EXTRACTION": parsed}
                else:
                    parsed = _fallback_to_a(json_a)
            except Exception:
                parsed = _fallback_to_a(json_a)

        # Canonicalize fields
        inner = _canonicalize(parsed.get("JSON_EXTRACTION", parsed))

        # Tạo json_id cho gold standard
        json_id = _generate_json_id(image_id, "gold")

        # Bổ sung metadata — ghi nhận nguồn từ cả A và B
        inner.setdefault("_metadata", {}).update({
            "image_id":           image_id,
            "json_id":            json_id,
            "gemma_model":       self.model_id,
            "source_claude":     True,
            "source_gpt4o":      True,
            "source_claude_id":  _get_json_id(json_a),
            "source_gpt4o_id":   _get_json_id(json_b),
            "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
            "gemma_raw":        raw[:300],
        })

        return {"JSON_EXTRACTION": inner}


# ═══════════════════════════════════════════════════════════════
#  Alias cho backward compatibility
# ═══════════════════════════════════════════════════════════════
class GemmaComparisonEngine(GemmaGoldStandardEngine):
    """Alias tương thích — code cũ gọi GemmaComparisonEngine."""
    pass


# ═══════════════════════════════════════════════════════════════
#  Standalone helpers (module-level)
# ═══════════════════════════════════════════════════════════════
def encode_image(image_path: str, max_size=(1024, 1024)) -> str:
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img.thumbnail(max_size, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _generate_json_id(image_id: str, source: str) -> str:
    safe_img = re.sub(r'[^\w]', '_', image_id)
    safe_src = re.sub(r'[^\w]', '_', source)
    ts       = time.strftime("%m%d%H%M")
    return f"JSON_{safe_img}_{safe_src}_{ts}"


def generate_image_id_fallback(image_path: str) -> str:
    """Fallback: tạo image_id khi không có thông tin đầy đủ."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    short    = basename[:40].strip()
    h        = hashlib.md5(image_path.encode()).hexdigest()[:4]
    safe     = re.sub(r'[^\w]', '_', short)
    return f"IMG_{safe}_{h}"


CANONICAL_KEYS = [
    "Category", "Distribution", "Lesion_Type",
    "Colour", "Shape", "Characteristics", "_metadata",
]


def _canonicalize(data: dict) -> dict:
    ordered = {}
    for key in CANONICAL_KEYS:
        if key in data:
            ordered[key] = data[key]
    return ordered


def _fallback_to_a(json_a: dict) -> dict:
    inner = json_a.get("JSON_EXTRACTION", json_a)
    return {"JSON_EXTRACTION": _canonicalize(inner)}


def _get_json_id(j: dict) -> str:
    """Trích json_id từ JSON dict (hỗ trợ cả wrapper và không wrapper)."""
    return (j.get("JSON_EXTRACTION", {}).get("_metadata", {}).get("json_id", "")
            or j.get("_metadata", {}).get("json_id", "")
            or "")


# ═══════════════════════════════════════════════════════════════
#  Convenience wrapper
# ═══════════════════════════════════════════════════════════════
def create_gold_standard(
    image_path: str,
    json_a: dict,
    json_b: dict,
    disease_name: str,
    disease_knowledge: str = "",
    image_id: str = None,
) -> dict:
    """Wrapper đơn giản hóa việc gọi Gemma."""
    engine = GemmaGoldStandardEngine()
    engine.load_model()
    if image_id is None:
        image_id = generate_image_id_fallback(image_path)
    return engine.create_gold_standard(
        image_path, json_a, json_b,
        disease_name, disease_knowledge, image_id,
    )


# ═══════════════════════════════════════════════════════════════
#  Standalone test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Test Gemma Gold Standard Engine...")
    print("Model:", GEMMA4_MODEL_ID)
    print("Chạy: python Phase_1/scripts/run_pipeline.py để merge thực tế.")
