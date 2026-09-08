import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd


REASONING_CATEGORY = "Lesion_Reasoning"
REQUIRED_BASE_COLUMNS = {"index", "question", "answer", "category", "type"}


def _load_table(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported table format: {path.suffix}")


def _write_table_atomic(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        if path.suffix.lower() == ".xlsx":
            frame.to_excel(temporary_path, index=False)
        elif path.suffix.lower() == ".tsv":
            frame.to_csv(temporary_path, sep="\t", index=False)
        else:
            raise ValueError(f"unsupported table format: {path.suffix}")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_unique_indices(frame, label):
    if "index" not in frame.columns:
        raise ValueError(f"{label} is missing the index column")
    if frame["index"].isna().any():
        raise ValueError(f"{label} contains missing indices")
    if frame["index"].duplicated().any():
        raise ValueError(f"{label} contains duplicate indices")


def prepare_reasoning_dataset(base_dataset, mini_dataset):
    base = _load_table(base_dataset)
    missing = sorted(REQUIRED_BASE_COLUMNS.difference(base.columns))
    if missing:
        raise ValueError(f"base dataset is missing columns: {missing}")
    _validate_unique_indices(base, "base dataset")

    reasoning = base[base["category"] == REASONING_CATEGORY].copy()
    if reasoning.empty:
        raise ValueError("base dataset contains no Lesion_Reasoning rows")
    _write_table_atomic(reasoning, mini_dataset)
    return len(reasoning)


def merge_reasoning_result(base_dataset, original_result, patch_result, backup_dir):
    base = _load_table(base_dataset)
    original = _load_table(original_result)
    patch = _load_table(patch_result)
    for frame, label in ((base, "base dataset"), (original, "original result"), (patch, "patch result")):
        _validate_unique_indices(frame, label)
    missing = sorted(REQUIRED_BASE_COLUMNS.difference(base.columns))
    if missing:
        raise ValueError(f"base dataset is missing columns: {missing}")
    if "prediction" not in original.columns or "prediction" not in patch.columns:
        raise ValueError("result files must contain the prediction column")

    reasoning = base[base["category"] == REASONING_CATEGORY].copy()
    expected = set(reasoning["index"])
    if not expected:
        raise ValueError("base dataset contains no Lesion_Reasoning rows")
    original_indices = set(original["index"])
    patch_predictions = patch[["index", "prediction"]].dropna(subset=["prediction"])
    patch_predictions = patch_predictions[
        patch_predictions["prediction"].astype(str).str.strip().ne("")
        & ~patch_predictions["prediction"].astype(str).str.contains(
            r"Failed to obtain answer|SKIP: Image not found", case=False, regex=True
        )
    ]
    available = set(patch_predictions["index"])
    missing_original = expected.difference(original_indices)
    missing_predictions = expected.difference(available)
    if missing_original:
        raise ValueError(f"original result is missing reasoning rows: {len(missing_original)}")
    if missing_predictions:
        raise ValueError(f"missing reasoning predictions: {len(missing_predictions)}")

    merged = original.set_index("index", drop=False)
    base_by_index = reasoning.set_index("index")
    patch_by_index = patch_predictions.set_index("index")
    for column in ("question", "answer", "category", "type"):
        merged.loc[list(expected), column] = base_by_index.loc[list(expected), column]
    merged.loc[list(expected), "prediction"] = patch_by_index.loc[list(expected), "prediction"]
    if "score" in merged.columns:
        merged.loc[list(expected), "score"] = pd.NA
    merged = merged.reset_index(drop=True)

    original_path = Path(original_result)
    backup_root = Path(backup_dir)
    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = backup_root / f"{original_path.stem}.{timestamp}{original_path.suffix}"
    shutil.copy2(original_path, backup)
    _write_table_atomic(merged, original_path)
    return backup


def main():
    parser = argparse.ArgumentParser(description="Prepare and merge DermNet Lesion_Reasoning patches")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base", required=True)
    prepare.add_argument("--mini", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--base", required=True)
    merge.add_argument("--original", required=True)
    merge.add_argument("--patch-result", required=True)
    merge.add_argument("--backup-dir", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        count = prepare_reasoning_dataset(args.base, args.mini)
        print(f"Prepared {count} Lesion_Reasoning rows: {args.mini}")
    else:
        backup = merge_reasoning_result(
            args.base, args.original, args.patch_result, args.backup_dir
        )
        print(f"Merged Lesion_Reasoning predictions; backup: {backup}")


if __name__ == "__main__":
    main()
