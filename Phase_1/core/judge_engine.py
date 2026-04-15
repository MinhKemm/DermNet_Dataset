"""
Judge Engine — Gemma 4 qua Google AI Studio (miễn phí)

Nhiệm vụ:
  1. Nhận JSON từ VLM1 (Claude Opus) và VLM2 (GPT-4o)
  2. Tính Jaccard Similarity giữa 2 JSON
  3. Nếu Jaccard >= threshold → dùng JSON VLM1 trực tiếp
  4. Nếu Jaccard <  threshold → gọi Gemma 4 hợp nhất & sinh JSON cuối cùng

Endpoint: Google AI Studio (OpenAI-compatible)
  https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
  ?key=GOOGLE_AI_STUDIO_KEY

Tách biệt hoàn toàn khỏi VLM Engine.
"""

import os
import json
import re
import time
import requests

try:
    from Phase_1.utils.metrics import calculate_jaccard
except ImportError:
    def calculate_jaccard(d1, d2): return 0.85


# ─────────────────────────────────────────────────────────────────
#  CẤU HÌNH Gemma 4 — Google AI Studio (miễn phí)
# ─────────────────────────────────────────────────────────────────
GOOGLE_AI_STUDIO_KEY = os.environ.get("GOOGLE_AI_STUDIO_KEY", "")

# Endpoint OpenAI-compatible của Google AI Studio
GEMMA4_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/openai/chat/completions"
)
# Model Gemma 4 trên AI Studio (tên chính xác lấy từ model list)
GEMMA4_MODEL_ID = "gemma-3-27b-it"   # ← đổi model_id trong settings.yaml nếu cần


# ─────────────────────────────────────────────────────────────────
#  PROMPTS cho Judge (Gemma 4)
# ─────────────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "Bạn là chuyên gia hội chẩn Da liễu AI cấp cao. "
    "Nhiệm vụ: hợp nhất dữ liệu từ hai mô hình VLM để tạo Gold Standard JSON.\n\n"
    "QUY TẮC CỐT LÕI:\n"
    "1. Output PHẢI là JSON duy nhất với root key 'JSON_EXTRACTION'.\n"
    "2. Trường 'Category': ghi chính xác tên bệnh được cung cấp.\n"
    "3. Các trường còn lại BẮT BUỘC là mảng chuỗi [] (không để trống).\n"
    "4. Hợp nhất:\n"
    "   - Từ đồng nghĩa → dùng thuật ngữ y khoa chuẩn.\n"
    "   - Gộp đặc điểm không mâu thuẫn.\n"
    "   - Loại bỏ 'Không quan sát rõ' nếu model kia nhìn thấy rõ.\n"
    "5. KHÔNG giải thích, KHÔNG sinh text ngoài block JSON."
)

JUDGE_USER_TEMPLATE = (
    "[DỮ LIỆU ĐẦU VÀO]\n"
    "- Jaccard Similarity: {j_score:.2f}\n"
    "- VLM1 (Claude Opus 4.6): {vlm1_json}\n"
    "- VLM2 (GPT-4o): {vlm2_json}\n\n"
    "Hợp nhất và trả về JSON ngay lập tức:"
)

JUDGE_FEW_SHOT = """
--- VÍ DỤ MINH HỌA ---

Ví dụ 1 (đồng thuận cao):
  VLM1: {"Category":"Nấm da ẩn danh","Lesion_Type":["Mảng hồng ban"],"Colour":["Hồng ban"]}
  VLM2: {"Category":"Nấm da ẩn danh","Lesion_Type":["Mảng hồng ban"],"Colour":["Đỏ hồng"]}
  JSON Output:
  ```json
  {{"JSON_EXTRACTION":{{
    "Category":"Nấm da ẩn danh",
    "Lesion_Type":["Mảng hồng ban"],
    "Colour":["Hồng ban","Đỏ hồng"],
    "Shape":[],
    "Distribution":[],
    "Characteristics":[]
  }}}}
  ```

Ví dụ 2 (mâu thuẫn):
  VLM1: {"Category":"Ban đỏ đa dạng","Lesion_Type":["Sẩn hồng ban"]}
  VLM2: {"Category":"Ban đỏ đa dạng","Lesion_Type":["Mảng đỏ","Mụn nước"]}
  JSON Output:
  ```json
  {{"JSON_EXTRACTION":{{
    "Category":"Ban đỏ đa dạng",
    "Lesion_Type":["Sẩn hồng ban","Mảng đỏ"],
    "Shape":[],
    "Distribution":[],
    "Characteristics":["Mụn nước"]
  }}}}
  ```

--> BÂY GIỜ LÀ LƯỢT CỦA BẠN:
"""


# ─────────────────────────────────────────────────────────────────
#  JudgeEngine
# ─────────────────────────────────────────────────────────────────
class JudgeEngine:
    def __init__(self, model_id: str = GEMMA4_MODEL_ID):
        self.model_id  = model_id
        self.base_url  = GEMMA4_BASE_URL
        self.api_key   = GOOGLE_AI_STUDIO_KEY
        self._session  = requests.Session()

    def flush_memory(self):
        print("[Judge/Gemma4] flush_memory — no GPU needed.")

    def load_model(self):
        print(f"[Judge] Model: {self.model_id}")
        print(f"[Judge] Backend: Google AI Studio (miễn phí)")
        if not self.api_key:
            print("[Judge] ⚠️ CẢNH BÁO: GOOGLE_AI_STUDIO_KEY chưa đặt trong .env!")

    # ──────────────────────────────────────────
    #  Core: run(vlm1_json, vlm2_json, disease_name)
    # ──────────────────────────────────────────
    def run(self, vlm1_json: dict, vlm2_json: dict,
            disease_name: str, jaccard_threshold: float = 0.85) -> dict:
        """
        Luồng Judge:
          1. Kiểm tra lỗi JSON → fallback nếu 1 trong 2 lỗi
          2. Tính Jaccard
          3. Jaccard >= threshold → dùng JSON VLM1 trực tiếp
          4. Jaccard <  threshold → gọi Gemma 4 hợp nhất
        """
        err1 = self._has_error(vlm1_json)
        err2 = self._has_error(vlm2_json)

        if err1 and not err2:
            result = self._wrap(vlm2_json, "fallback_vlm2_only", jaccard=0.0)
            print(f"[Judge] ⚠️ VLM1 lỗi → dùng VLM2 trực tiếp.")
            return result

        if err2 and not err1:
            result = self._wrap(vlm1_json, "fallback_vlm1_only", jaccard=0.0)
            print(f"[Judge] ⚠️ VLM2 lỗi → dùng VLM1 trực tiếp.")
            return result

        if err1 and err2:
            return {
                "error": "Cả hai VLM đều lỗi.",
                "JSON_EXTRACTION": {"Category": disease_name}
            }

        # Tính Jaccard
        j_score = calculate_jaccard(vlm1_json, vlm2_json)
        print(f"[Judge] Jaccard Score: {j_score:.4f}  (threshold: {jaccard_threshold})")

        # Threshold: chấp nhận trực tiếp
        if j_score >= jaccard_threshold:
            result = self._wrap(vlm1_json, "consensus_direct", jaccard=j_score)
            print(f"[Judge] ✅ Jaccard >= threshold → dùng VLM1 trực tiếp.")
            return result

        # Jaccard thấp → gọi Gemma 4 hợp nhất
        print(f"[Judge] 🔄 Jaccard < threshold → gọi Gemma 4 hợp nhất...")
        return self._run_gemma_merge(vlm1_json, vlm2_json, j_score, disease_name)

    # ──────────────────────────────────────────
    #  Gemma 4 merge
    # ──────────────────────────────────────────
    def _run_gemma_merge(self, vlm1, vlm2, j_score, disease_name) -> dict:
        try:
            raw = self._call_gemma(vlm1, vlm2, j_score, disease_name)
            parsed = self.extract_json(raw)
            parsed["JSON_EXTRACTION"]["_metadata"] = {
                "judge_model":   self.model_id,
                "judge_backend": "google-ai-studio",
                "input_jaccard": round(j_score, 4),
                "status":        "gemma4_merged",
                "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"[Judge/Gemma4] ✅ JSON parse thành công.")
            return parsed
        except Exception as e:
            print(f"[-] [Judge] Gemma merge thất bại: {e} → fallback VLM1.")
            return self._wrap(vlm1, "gemma_merge_failed", j_score)

    def _call_gemma(self, vlm1, vlm2, j_score, disease_name) -> str:
        """Gọi Gemma 4 qua Google AI Studio (OpenAI-compatible endpoint)."""
        if not self.api_key:
            raise RuntimeError(
                "[Judge] GOOGLE_AI_STUDIO_KEY chưa đặt trong .env. "
                "Lấy key tại: https://aistudio.google.com/app/apikey"
            )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    JUDGE_USER_TEMPLATE.format(
                        j_score   = j_score,
                        vlm1_json = json.dumps(vlm1, ensure_ascii=False),
                        vlm2_json = json.dumps(vlm2, ensure_ascii=False),
                    )
                    + JUDGE_FEW_SHOT
                ),
            },
        ]

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.05,
            "max_tokens": 2048,
        }

        # Google AI Studio: key là query param
        url = f"{self.base_url}?key={self.api_key}"

        print(f"[Judge] Đang gọi {self.model_id} qua Google AI Studio...")
        resp = self._session.post(url, json=payload, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────
    def _has_error(self, data: dict) -> bool:
        return "error" in data or not data

    def _wrap(self, data: dict, status: str, jaccard: float) -> dict:
        if "JSON_EXTRACTION" not in data:
            data = {"JSON_EXTRACTION": data}
        data["JSON_EXTRACTION"]["_metadata"] = {
            "judge_model":    self.model_id,
            "judge_backend":  "none_direct",
            "input_jaccard":  round(jaccard, 4),
            "status":         status,
            "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return data

    def extract_json(self, text: str) -> dict:
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
            print(f"[-] [Judge] JSON parse failed: {e}")
            return {"error": f"JSON parse failed: {e}", "raw": text[:300]}


# ─────────────────────────────────────────────────────────────────
#  Convenience wrapper
# ─────────────────────────────────────────────────────────────────
def judge_merge(vlm1_json: dict, vlm2_json: dict,
                disease_name: str, threshold: float = 0.85) -> dict:
    judge = JudgeEngine()
    judge.load_model()
    return judge.run(vlm1_json, vlm2_json, disease_name, threshold)


if __name__ == "__main__":
    print("Test Judge Engine (Gemma 4 — Google AI Studio)...")
    judge = JudgeEngine()
    judge.load_model()
    result = judge.run(
        vlm1_json={
            "Category": "Ban đỏ đa dạng",
            "Lesion_Type": ["Tổn thương hình bia bắn"],
            "Colour": ["Đỏ", "Trung tâm sẫm màu"],
        },
        vlm2_json={
            "Category": "Ban đỏ đa dạng",
            "Lesion_Type": ["Sẩn hồng ban", "Mụn nước"],
            "Colour": ["Hồng đỏ", "Vòng đồng tâm"],
        },
        disease_name="Ban đỏ đa dạng",
        jaccard_threshold=0.85,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))