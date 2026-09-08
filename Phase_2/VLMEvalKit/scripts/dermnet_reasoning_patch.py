import argparse
import json
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile
import unicodedata

import pandas as pd


REASONING_CATEGORY = "Lesion_Reasoning"
REQUIRED_BASE_COLUMNS = {"index", "question", "answer", "category", "type"}


def _approved_question_change(index, expected, source):
    audit = Path(__file__).with_name('dataset_repairs.json')
    if not audit.exists():
        return False
    normalize = lambda value: ' '.join(str(value).split())
    return any(
        item['index'] == index
        and _image_identity(item['image_path']) == _image_identity(expected.image_path)
        and normalize(item['before']) == normalize(source.question)
        and normalize(item['after']) == normalize(expected.question)
        for item in json.loads(audit.read_text(encoding='utf-8'))
    )


def selected_patch_rows(base, original):
    validate_source_frames(base, original)
    source = original.set_index('index').loc[base['index']]
    changed = base['category'].eq(REASONING_CATEGORY).to_numpy()
    for column in ('question', 'answer'):
        changed |= (base[column].astype(str).to_numpy() != source[column].astype(str).to_numpy())
    return base.loc[changed].copy()


def _image_identity(value):
    if pd.isna(value):
        raise ValueError("missing image_path")
    text = unicodedata.normalize("NFC", str(value).replace("\\", "/"))
    if "/images/" not in text:
        raise ValueError(f"image_path has no unambiguous /images/ root: {text}")
    return text.split("/images/", 1)[1].casefold()


def validate_source_frames(base, original, allow_reasoning_changes=True):
    for frame, label in ((base, "base dataset"), (original, "source result")):
        _validate_unique_indices(frame, label)
        missing = (REQUIRED_BASE_COLUMNS | {"image_path"}).difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
    expected = base.set_index("index")
    source = original.set_index("index")
    if not set(expected.index).issubset(source.index):
        raise ValueError("source result missing dataset indices")
    source = source.loc[expected.index]
    wrong = expected.image_path.map(_image_identity) != source.image_path.map(_image_identity)
    if wrong.any():
        raise ValueError(f"source image mismatch at {int(wrong.sum())} indices")
    unchanged = expected.category != REASONING_CATEGORY if allow_reasoning_changes else pd.Series(True, index=expected.index)
    for column in ("question", "category", "type"):
        left = expected.loc[unchanged, column].astype(str).map(lambda s: " ".join(s.split()))
        right = source.loc[unchanged, column].astype(str).map(lambda s: " ".join(s.split()))
        mismatch = left != right
        if column == 'question' and allow_reasoning_changes:
            for index in mismatch[mismatch].index:
                if _approved_question_change(index, expected.loc[index], source.loc[index]):
                    mismatch.loc[index] = False
        if mismatch.any():
            raise ValueError(f"source {column} mismatch outside permitted reasoning changes")


def validate_source(base_dataset, original_result):
    base, original = _load_table(base_dataset), _load_table(original_result)
    if "prediction" not in original:
        raise ValueError("source result missing prediction")
    validate_source_frames(base, original)
    return len(base)


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


def prepare_reasoning_dataset(base_dataset, mini_dataset, original_result=None):
    base = _load_table(base_dataset)
    missing = sorted(REQUIRED_BASE_COLUMNS.difference(base.columns))
    if missing:
        raise ValueError(f"base dataset is missing columns: {missing}")
    _validate_unique_indices(base, "base dataset")

    reasoning = base[base["category"] == REASONING_CATEGORY].copy()
    if original_result:
        reasoning = selected_patch_rows(base, _load_table(original_result))
    if reasoning.empty:
        raise ValueError("base dataset contains no Lesion_Reasoning rows")
    _write_table_atomic(reasoning, mini_dataset)
    return len(reasoning)


def merge_reasoning_result(base_dataset, original_result, patch_result, backup_dir, output_result=None):
    base = _load_table(base_dataset)
    original = _load_table(original_result)
    patch = _load_table(patch_result)
    validate_source_frames(base, original)
    reasoning_base = selected_patch_rows(base, original)
    validate_source_frames(reasoning_base, patch, allow_reasoning_changes=False)
    for frame, label in ((base, "base dataset"), (original, "original result"), (patch, "patch result")):
        _validate_unique_indices(frame, label)
    missing = sorted(REQUIRED_BASE_COLUMNS.difference(base.columns))
    if missing:
        raise ValueError(f"base dataset is missing columns: {missing}")
    if "prediction" not in original.columns or "prediction" not in patch.columns:
        raise ValueError("result files must contain the prediction column")

    reasoning = reasoning_base.copy()
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
    _write_table_atomic(merged, Path(output_result) if output_result else original_path)
    return backup


def main():
    parser = argparse.ArgumentParser(description="Prepare and merge DermNet Lesion_Reasoning patches")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base", required=True)
    prepare.add_argument("--mini", required=True)
    prepare.add_argument("--original")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--base", required=True)
    merge.add_argument("--original", required=True)
    merge.add_argument("--patch-result", required=True)
    merge.add_argument("--backup-dir", required=True)
    merge.add_argument("--output", help="Write merged result separately, preserving the original input")
    check = subparsers.add_parser("check")
    check.add_argument("--base", required=True)
    check.add_argument("--original", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        count = prepare_reasoning_dataset(args.base, args.mini, args.original)
        print(f"Prepared {count} Lesion_Reasoning rows: {args.mini}")
    elif args.command == "check":
        print(f"Compatible source: {validate_source(args.base, args.original)} rows")
    else:
        backup = merge_reasoning_result(
            args.base, args.original, args.patch_result, args.backup_dir, args.output
        )
        print(f"Merged Lesion_Reasoning predictions; backup: {backup}")


if __name__ == "__main__":
    main()
