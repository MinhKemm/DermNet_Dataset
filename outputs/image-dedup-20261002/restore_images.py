"""Restore original image paths from verified backup. Dry run unless --apply."""
from pathlib import Path
import argparse, hashlib, json, zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true", help="Restore original image files")
args = parser.parse_args()
out = Path(__file__).resolve().parent
root = out.parents[1]
image_root = (root / "dermnet-output/images").resolve()
rows = json.loads((out / "removal_journal.json").read_text(encoding="utf-8"))
with zipfile.ZipFile(out / "removed_images_backup.zip") as archive:
    pending = []
    for row in rows:
        target = (image_root / row["remove"]).resolve()
        if not target.is_relative_to(image_root):
            raise ValueError("Path outside image directory")
        data = archive.read(row["backup_member"])
        expected = row["removed_sha256"]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Backup hash mismatch: " + row["remove"])
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("Existing file differs; refusing overwrite: " + str(target))
        else:
            pending.append((target, data))
    print(f"Verified backup; {len(pending)} original files can be restored.")
    if args.apply:
        for target, data in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        print("Restored original image paths. Duplicate files are restored too.")
    else:
        print("Dry run only. Pass --apply to restore. The pending review copy is preserved.")
