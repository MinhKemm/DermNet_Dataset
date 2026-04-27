"""
RegistryManager — Quản lý tiến độ xử lý ảnh trong pipeline (Luồng GPT Đơn)

Tự động:
  • Ghi log ảnh nào đã chạy, ảnh nào chưa
  • Tránh chạy lại ảnh đã xong
  • Lưu trạng thái mỗi phase (PENDING / P1_OK / P2_OK / ERROR)
  • Tự động quét và chạy lại các file bị ERROR
"""

import os
import csv
import re

STATUS_PENDING          = "PENDING"           # chưa chạy gì
STATUS_P1_RUNNING       = "P1_RUNNING"        # đang chạy Phase 1
STATUS_P1_OK            = "P1_OK"             # Phase 1 xong
STATUS_P2_RUNNING       = "P2_RUNNING"        # đang chạy Phase 2
STATUS_P2_OK            = "P2_OK"             # Phase 2 xong (Hoàn thành)
STATUS_ERROR            = "ERROR"             # có lỗi, sẽ được retry

COLUMNS = [
    "image_path", "disease_name", "disease_folder",
    "status",
    "phase1_status", "phase1_raw",
    "phase2_status", "phase2_raw",
    "error_log"
]

# ═══════════════════════════════════════════════════════════════
#  RegistryManager
# ═══════════════════════════════════════════════════════════════
class RegistryManager:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._ensure_exists()

    def _ensure_exists(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
            print(f"[Registry] ✅ Tạo mới: {self.csv_path}")

    def _rows(self) -> list[dict]:
        rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def _save(self, rows: list[dict]):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def discover_dataset(self, images_dir: str, contents_dir: str):
        if not os.path.exists(images_dir):
            print(f"[Registry] ⚠️ Không tìm thấy: {images_dir}")
            return 0

        existing = {row["image_path"] for row in self._rows()}
        new_count = 0

        for disease_folder in sorted(os.listdir(images_dir)):
            disease_path = os.path.join(images_dir, disease_folder)
            if not os.path.isdir(disease_path):
                continue

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
                    "disease_name":     self._extract_disease_name(knowledge_name) if knowledge_name else disease_folder,
                    "disease_folder":   disease_folder,
                    "status":            STATUS_PENDING,
                    "phase1_status":     "",
                    "phase1_raw":        "",
                    "phase2_status":     "",
                    "phase2_raw":        "",
                    "error_log":         ""
                })
                new_count += 1

        if new_count:
            print(f"[Registry] ✅ Đã thêm {new_count} ảnh mới vào registry")
        else:
            print(f"[Registry] ℹ️ Không có ảnh mới — registry đã đầy")

        return new_count

    def _find_knowledge(self, contents_dir: str, disease_folder: str):
        if not os.path.exists(contents_dir):
            return None
        # Format chuẩn của file txt
        expected = f"Toàn bộ nội dung - {disease_folder}.txt"
        path = os.path.join(contents_dir, expected)
        return path if os.path.exists(path) else None

    def _extract_disease_name(self, knowledge_filename: str) -> str:
        name = re.sub(r"^Toàn bộ nội dung \-\s*", "", knowledge_filename)
        name = re.sub(r"\.(txt|TXT)$", "", name).strip()
        return name

    def _add_entry(self, data: dict):
        rows = self._rows()
        rows.append(data)
        self._save(rows)

    def get_pending(self) -> list[dict]:
        """Lấy tất cả ảnh CHƯA đạt STATUS_P2_OK (bao gồm cả ERROR và PENDING)."""
        return [
            row for row in self._rows()
            if row["status"] in [STATUS_PENDING, STATUS_ERROR, STATUS_P1_OK]
        ]

    def update_status(self, image_rel_path: str, status: str, extra: dict | None = None):
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

    def update_phase(self, image_rel_path: str, phase: str, phase_status: str, raw_text: str = "", error: str = ""):
        rows = self._rows()
        for row in rows:
            if row["image_path"] != image_rel_path:
                continue

            if phase == "phase1":
                row["phase1_status"] = phase_status
                row["phase1_raw"]    = raw_text[:2000]
                if phase_status == STATUS_ERROR:
                    row["status"] = STATUS_ERROR
                    row["error_log"] = error

            elif phase == "phase2":
                row["phase2_status"] = phase_status
                row["phase2_raw"]    = raw_text[:2000]
                if phase_status == STATUS_P2_OK:
                    row["status"] = STATUS_P2_OK
                elif phase_status == STATUS_ERROR:
                    row["status"] = STATUS_ERROR
                    row["error_log"] = error

            break
        self._save(rows)