"""
RegistryManager — Quản lý tiến độ xử lý ảnh trong pipeline

Tự động:
  • Ghi log ảnh nào đã chạy, ảnh nào chưa
  • Tránh chạy lại ảnh đã xong
  • Lưu trạng thái mỗi phase (P1_PENDING / P1_OK / P2_OK / COMPARED)
  • Hỗ trợ resume giữa chừng

CSV columns:
  image_path, disease_name, status, model_claude, model_gpt4o,
  phase1_status, phase2_status, gemma_verdict, gemma_reason,
  jaccard_score, error_log, started_at, finished_at
"""

import os
import csv
import time
import json
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
#  Status constants
# ═══════════════════════════════════════════════════════════════
STATUS_PENDING          = "PENDING"           # chưa chạy gì
STATUS_P1_RUNNING       = "P1_RUNNING"        # đang chạy Phase 1
STATUS_P1_OK            = "P1_OK"             # Phase 1 xong
STATUS_P2_RUNNING       = "P2_RUNNING"        # đang chạy Phase 2
STATUS_P2_OK            = "P2_OK"             # Phase 2 xong
STATUS_GEMMA_RUNNING    = "GEMMA_RUNNING"     # đang chạy Gemma so sánh
STATUS_GEMMA_OK         = "GEMMA_OK"          # Gemma so sánh xong
STATUS_ERROR            = "ERROR"             # có lỗi

COLUMNS = [
    "image_path", "disease_name", "disease_folder",
    "status",
    # Phase 1 (Claude)
    "phase1_claude_status", "phase1_claude_raw",
    # Phase 2 (Claude)
    "phase2_claude_status", "phase2_claude_raw",
    # GPT-4o JSON (bạn bè chạy, đặt vào gpt4o_outputs/)
    "phase1_gpt4o_status", "phase1_gpt4o_raw",
    "phase2_gpt4o_status", "phase2_gpt4o_raw",
    # Gemma comparison
    "gemma_verdict", "gemma_reason", "gemma_model",
    # Metadata
    "error_log", "started_at", "finished_at",
]


# ═══════════════════════════════════════════════════════════════
#  RegistryManager
# ═══════════════════════════════════════════════════════════════
class RegistryManager:
    """
    Quản lý master_registry.csv — theo dõi tiến độ pipeline.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._ensure_exists()

    # ──────────────────────────────────────────────────────────
    #  Initialize
    # ──────────────────────────────────────────────────────────
    def _ensure_exists(self):
        """Tạo CSV nếu chưa tồn tại."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
            print(f"[Registry] ✅ Tạo mới: {self.csv_path}")

    def _rows(self) -> list[dict]:
        """Đọc toàn bộ CSV thành list dict."""
        rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def _save(self, rows: list[dict]):
        """Ghi lại toàn bộ CSV."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    # ──────────────────────────────────────────────────────────
    #  Discovery — quét dataset tạo registry ban đầu
    # ──────────────────────────────────────────────────────────
    def discover_dataset(self, images_dir: str, contents_dir: str):
        """
        Quét dermnet-output/ → thêm entry mới (chưa có trong CSV).
        Chỉ thêm ảnh mới, không ghi đè entry đã có.
        """
        if not os.path.exists(images_dir):
            print(f"[Registry] ⚠️ Không tìm thấy: {images_dir}")
            return 0

        existing = {row["image_path"] for row in self._rows()}
        new_count = 0

        for disease_folder in sorted(os.listdir(images_dir)):
            disease_path = os.path.join(images_dir, disease_folder)
            if not os.path.isdir(disease_path):
                continue

            # Tìm kiến thức bệnh
            knowledge_path = self._find_knowledge(contents_dir, disease_folder)
            knowledge_name = os.path.basename(knowledge_path) if knowledge_path else ""

            for img_file in sorted(os.listdir(disease_path)):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_rel = os.path.join(disease_folder, img_file)
                if img_rel in existing:
                    continue

                self._add_entry({
                    "image_path":       img_rel,
                    "disease_name":     self._extract_disease_name(knowledge_name),
                    "disease_folder":   disease_folder,
                    "status":            STATUS_PENDING,
                    "phase1_claude_status": "",
                    "phase1_claude_raw":  "",
                    "phase2_claude_status": "",
                    "phase2_claude_raw":  "",
                    "phase1_gpt4o_status": "",
                    "phase1_gpt4o_raw":   "",
                    "phase2_gpt4o_status": "",
                    "phase2_gpt4o_raw":   "",
                    "gemma_verdict":      "",
                    "gemma_reason":       "",
                    "gemma_model":        "",
                    "error_log":          "",
                    "started_at":          "",
                    "finished_at":         "",
                })
                new_count += 1

        if new_count:
            print(f"[Registry] ✅ Đã thêm {new_count} ảnh mới vào registry")
        else:
            print(f"[Registry] ℹ️ Không có ảnh mới — registry đã đầy")

        return new_count

    def _find_knowledge(self, contents_dir: str, disease_folder: str):
        """Tìm file kiến thức .txt trong contents/."""
        if not os.path.exists(contents_dir):
            return None
        target = disease_folder.lower().strip()
        for fname in sorted(os.listdir(contents_dir)):
            if not fname.lower().endswith(".txt"):
                continue
            part = re.sub(r"^.*?\-\s*", "", fname)
            part = re.sub(r"\.txt$", "", part, flags=re.IGNORECASE).strip()
            if part.lower() == target or target in part.lower():
                return os.path.join(contents_dir, fname)
        return None

    def _extract_disease_name(self, knowledge_filename: str) -> str:
        import re as _re
        name = _re.sub(r"^.*?\-\s*", "", knowledge_filename)
        name = _re.sub(r"\.(txt|TXT)$", "", name).strip()
        return name

    # ──────────────────────────────────────────────────────────
    #  CRUD operations
    # ──────────────────────────────────────────────────────────
    def _add_entry(self, data: dict):
        rows = self._rows()
        rows.append(data)
        self._save(rows)

    def get_entry(self, image_rel_path: str) -> dict | None:
        """Lấy 1 entry theo image_path."""
        for row in self._rows():
            if row["image_path"] == image_rel_path:
                return row
        return None

    def get_pending(self) -> list[dict]:
        """Lấy danh sách ảnh chưa hoàn thành (status != *_OK & != GEMMA_OK)."""
        return [
            row for row in self._rows()
            if row["status"] not in (
                STATUS_P1_OK, STATUS_P2_OK, STATUS_GEMMA_OK, STATUS_ERROR
            )
        ]

    def get_completed(self) -> list[dict]:
        """Lấy danh sách ảnh đã hoàn thành."""
        return [
            row for row in self._rows()
            if row["status"] in (STATUS_P1_OK, STATUS_P2_OK, STATUS_GEMMA_OK)
        ]

    def update_status(self, image_rel_path: str,
                      status: str, extra: dict | None = None):
        """Cập nhật status + optional extra fields."""
        rows = self._rows()
        for row in rows:
            if row["image_path"] == image_rel_path:
                row["status"] = status
                if extra:
                    for k, v in extra.items():
                        if k in COLUMNS:
                            row[k] = v
                break
        self._save(rows)

    def update_phase(self, image_rel_path: str,
                     phase: str,  # "claude_p1" | "claude_p2" | "gemma"
                     phase_status: str,
                     raw_text: str = "",
                     verdict: str = "",
                     reason: str = "",
                     error: str = ""):
        """
        Cập nhật chi tiết từng phase.
        phase = "claude_p1" | "claude_p2" | "gemma"
        """
        rows = self._rows()
        for row in rows:
            if row["image_path"] != image_rel_path:
                continue

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if phase == "claude_p1":
                row["phase1_claude_status"] = phase_status
                row["phase1_claude_raw"]    = raw_text[:2000]
                if phase_status == STATUS_ERROR:
                    row["status"] = STATUS_ERROR
                    row["error_log"] = error

            elif phase == "claude_p2":
                row["phase2_claude_status"] = phase_status
                row["phase2_claude_raw"]    = raw_text[:2000]
                if phase_status == STATUS_P2_OK:
                    row["status"] = STATUS_P2_OK

            elif phase == "gemma":
                row["gemma_verdict"]   = verdict
                row["gemma_reason"]    = reason[:500]
                row["gemma_model"]     = "gemma-3-27b-it"
                if verdict:
                    row["status"] = STATUS_GEMMA_OK

            if phase_status in (STATUS_P1_OK, STATUS_P2_OK, STATUS_GEMMA_OK):
                row["finished_at"] = now

            break

        self._save(rows)

    # ──────────────────────────────────────────────────────────
    #  Stats
    # ──────────────────────────────────────────────────────────
    def summary(self) -> dict:
        rows = self._rows()
        total = len(rows)
        counts = {}
        for row in rows:
            s = row["status"] or STATUS_PENDING
            counts[s] = counts.get(s, 0) + 1
        return {
            "total":    total,
            "counts":   counts,
            "pending":  sum(1 for r in rows if r["status"] in (STATUS_PENDING, "", None)),
            "completed": sum(1 for r in rows if r["status"] in (STATUS_P2_OK, STATUS_GEMMA_OK)),
        }


# ═══════════════════════════════════════════════════════════════
#  CLI quick test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys, re
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    reg_path = os.path.join(project_root, "Phase_1", "master_registry.csv")

    print(f"Registry path: {reg_path}")
    reg = RegistryManager(reg_path)

    # Discover dataset
    img_dir = os.path.join(project_root, "dermnet-output", "images")
    txt_dir = os.path.join(project_root, "dermnet-output", "contents")
    reg.discover_dataset(img_dir, txt_dir)

    # Stats
    stats = reg.summary()
    print(f"\n📋 REGISTRY SUMMARY:")
    print(f"  Total  images : {stats['total']}")
    print(f"  Pending       : {stats['pending']}")
    print(f"  Completed     : {stats['completed']}")
    print(f"  By status     : {stats['counts']}")