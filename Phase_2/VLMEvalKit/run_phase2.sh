#!/usr/bin/env bash

set -Eeuo pipefail

# DermNet bilingual benchmark runner.
# Fresh run: bash run_phase2.sh all
# Continue:  bash run_phase2.sh resume

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"
SOURCE_DATA_DIR="$SCRIPT_DIR/LMUData"
export LMUData="$SOURCE_DATA_DIR"

STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/outputs/.phase2-runner}"
COMPLETED_DIR="$STATE_DIR/completed"
LOG_DIR="$STATE_DIR/logs"
LOCK_DIR="$STATE_DIR/lock"
RUNTIME_DATA_DIR="$STATE_DIR/datasets"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_QWEN="${PYTHON_QWEN:-$PYTHON_BIN}"
PYTHON_LEGACY="${PYTHON_LEGACY:-$PYTHON_BIN}"
PYTHON_DEEPSEEK="${PYTHON_DEEPSEEK:-$PYTHON_BIN}"
PYTHON_EXE="$PYTHON_BIN"

DRY_RUN="${DRY_RUN:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
MAX_JOB_RETRIES="${MAX_JOB_RETRIES:-2}"
MODEL_PROFILE="${MODEL_PROFILE:-auto}"
MISSING_IMAGE_POLICY="${MISSING_IMAGE_POLICY:-skip}"

# Override these values to test or force a profile without changing this file.
GPU_COUNT="${GPU_COUNT:-}"
GPU_MAX_VRAM_GB="${GPU_MAX_VRAM_GB:-}"
GPU_TOTAL_VRAM_GB="${GPU_TOTAL_VRAM_GB:-}"

DATASETS=(
    "DermNet_Val_4k-2_mac_relative"
    "DermNet_Test_mac_relative"
    "DermNet_Val_4k_en"
    "DermNet_Test_1of3_en"
)
MODELS=()
SKIPPED_MODELS=()
DEEPSEEK_VARIANT=""

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
job_id() { printf '%s__%s' "$1" "$2" | tr -c 'A-Za-z0-9_.-' '_'; }

usage() {
    cat <<'EOF'
DermNet QA - one-command bilingual server runner

Usage:
  bash run_phase2.sh all
  bash run_phase2.sh resume
  bash run_phase2.sh status
  bash run_phase2.sh plan
  bash run_phase2.sh full <model_name> <dataset_name>

The all/resume commands run both Vietnamese and English Val/Test datasets.
The auto profile selects only models that fit the detected NVIDIA VRAM and
chooses deepseek_vl2, deepseek_vl2_int8, or deepseek_vl2_int4 automatically.

Useful overrides:
  HF_TOKEN=...                 Hugging Face token (never put it in this file).
  MODEL_PROFILE=auto           auto | full | balanced | minimal
  PYTHON_BIN=python3           Python environment prepared by the server owner.
  PYTHON_QWEN=python3          Optional Python executable for Qwen models.
  PYTHON_LEGACY=python3        Optional Python executable for LLaVA/Vintern.
  PYTHON_DEEPSEEK=python3      Optional Python executable for DeepSeek-VL2.
  MAX_JOB_RETRIES=2            Attempts per model/dataset job.
  MISSING_IMAGE_POLICY=skip     skip | fail; original TSV files are never edited.
  GPU_MAX_VRAM_GB=80           Override detected largest-GPU VRAM.
  GPU_TOTAL_VRAM_GB=160        Override detected aggregate VRAM.
  DRY_RUN=1                    Validate and print commands without inference.

Examples:
  bash run_phase2.sh all
  bash run_phase2.sh resume
  MODEL_PROFILE=full bash run_phase2.sh plan
EOF
}

detect_hardware() {
    if [[ -n "$GPU_COUNT" && -n "$GPU_MAX_VRAM_GB" && -n "$GPU_TOTAL_VRAM_GB" ]]; then
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        if [[ "$DRY_RUN" == '1' ]]; then
            GPU_COUNT="${GPU_COUNT:-1}"
            GPU_MAX_VRAM_GB="${GPU_MAX_VRAM_GB:-80}"
            GPU_TOTAL_VRAM_GB="${GPU_TOTAL_VRAM_GB:-80}"
            return
        fi
        [[ "$REQUIRE_GPU" == '0' ]] || die 'nvidia-smi was not found; an NVIDIA/CUDA server is required.'
        GPU_COUNT="${GPU_COUNT:-0}"
        GPU_MAX_VRAM_GB="${GPU_MAX_VRAM_GB:-0}"
        GPU_TOTAL_VRAM_GB="${GPU_TOTAL_VRAM_GB:-0}"
        return
    fi

    local values
    values="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)" \
        || die 'Could not read GPU memory with nvidia-smi.'
    [[ -n "$values" ]] || die 'No NVIDIA GPU was detected.'

    read -r GPU_COUNT GPU_MAX_VRAM_GB GPU_TOTAL_VRAM_GB < <(
        awk '{ count += 1; total += $1; if ($1 > max) max = $1 }
             END { printf "%d %d %d\n", count, int(max / 1024), int(total / 1024) }' <<<"$values"
    )
}

add_model_if_fit() {
    local model="$1" required_max="$2" required_total="$3"
    if (( GPU_MAX_VRAM_GB >= required_max && GPU_TOTAL_VRAM_GB >= required_total )); then
        MODELS+=("$model")
    else
        SKIPPED_MODELS+=("$model (needs max>=${required_max}GB,total>=${required_total}GB)")
    fi
}

select_models() {
    detect_hardware
    MODELS=()
    SKIPPED_MODELS=()

    case "$MODEL_PROFILE" in
        full)
            MODELS=(
                "Qwen3.5-35B-A3B" "Qwen3-VL-8B-Instruct" "LLaVA-med-v1.5-7B"
                "Vintern-1B-v2" "Vintern-3B-beta" "deepseek_vl2_tiny"
                "deepseek_vl2_small" "deepseek_vl2"
            )
            DEEPSEEK_VARIANT='deepseek_vl2'
            ;;
        balanced)
            MODELS=(
                "Qwen3-VL-8B-Instruct" "LLaVA-med-v1.5-7B" "Vintern-1B-v2"
                "Vintern-3B-beta" "deepseek_vl2_tiny" "deepseek_vl2_small"
                "deepseek_vl2_int8"
            )
            DEEPSEEK_VARIANT='deepseek_vl2_int8'
            ;;
        minimal)
            MODELS=("Vintern-1B-v2" "Vintern-3B-beta" "deepseek_vl2_tiny" "deepseek_vl2_int4")
            DEEPSEEK_VARIANT='deepseek_vl2_int4'
            ;;
        auto)
            add_model_if_fit 'Vintern-1B-v2' 8 8
            add_model_if_fit 'Vintern-3B-beta' 12 12
            add_model_if_fit 'Qwen3-VL-8B-Instruct' 20 24
            add_model_if_fit 'LLaVA-med-v1.5-7B' 20 24
            add_model_if_fit 'Qwen3.5-35B-A3B' 24 72
            add_model_if_fit 'deepseek_vl2_tiny' 12 12
            add_model_if_fit 'deepseek_vl2_small' 36 36

            if (( GPU_MAX_VRAM_GB >= 64 )); then
                DEEPSEEK_VARIANT='deepseek_vl2'
            elif (( GPU_MAX_VRAM_GB >= 36 )); then
                DEEPSEEK_VARIANT='deepseek_vl2_int8'
            elif (( GPU_MAX_VRAM_GB >= 22 )); then
                DEEPSEEK_VARIANT='deepseek_vl2_int4'
            else
                DEEPSEEK_VARIANT=''
                SKIPPED_MODELS+=("deepseek_vl2 scalable variant (needs max>=22GB)")
            fi
            [[ -z "$DEEPSEEK_VARIANT" ]] || MODELS+=("$DEEPSEEK_VARIANT")
            ;;
        *) die 'MODEL_PROFILE must be auto, full, balanced, or minimal.' ;;
    esac

    (( ${#MODELS[@]} > 0 )) || die 'No configured model fits this server.'
}

show_plan() {
    select_models
    local number=0 model dataset
    printf 'Hardware: %s GPU(s), largest=%sGB, total=%sGB; profile=%s\n' \
        "$GPU_COUNT" "$GPU_MAX_VRAM_GB" "$GPU_TOTAL_VRAM_GB" "$MODEL_PROFILE"
    printf 'Execution plan: %s models x %s bilingual datasets = %s jobs\n' \
        "${#MODELS[@]}" "${#DATASETS[@]}" "$(( ${#MODELS[@]} * ${#DATASETS[@]} ))"
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            number=$((number + 1))
            printf '%2d. %-25s %s\n' "$number" "$model" "$dataset"
        done
    done
    if (( ${#SKIPPED_MODELS[@]} > 0 )); then
        printf 'Skipped by auto VRAM policy:\n'
        printf '  - %s\n' "${SKIPPED_MODELS[@]}"
    fi
}

validate_static_files() {
    [[ -f "$SCRIPT_DIR/run.py" ]] || die 'Missing run.py.'
    [[ -f "$SCRIPT_DIR/requirements.txt" ]] || die 'Missing requirements.txt.'
    [[ -f "$SCRIPT_DIR/vlmeval/config.py" ]] || die 'Missing vlmeval/config.py.'
    [[ "$MAX_JOB_RETRIES" =~ ^[1-9][0-9]*$ ]] || die 'MAX_JOB_RETRIES must be a positive integer.'

    local dataset model
    for dataset in "${DATASETS[@]}"; do
        [[ -f "$SOURCE_DATA_DIR/$dataset.tsv" ]] || die "Missing LMUData/$dataset.tsv"
    done
    [[ "$MISSING_IMAGE_POLICY" == 'skip' || "$MISSING_IMAGE_POLICY" == 'fail' ]] \
        || die 'MISSING_IMAGE_POLICY must be skip or fail.'
    for model in "${MODELS[@]}"; do
        grep -Fq -- "\"$model\"" "$SCRIPT_DIR/vlmeval/config.py" \
            || die "Model is not registered: $model"
    done
}

prepare_datasets() {
    [[ "$DRY_RUN" == '1' ]] && return
    mkdir -p "$RUNTIME_DATA_DIR"
    "$PYTHON_EXE" - "$SOURCE_DATA_DIR" "$RUNTIME_DATA_DIR" "$SCRIPT_DIR" \
        "$MISSING_IMAGE_POLICY" "${DATASETS[@]}" <<'PY'
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd

source_dir = Path(sys.argv[1]).resolve()
runtime_dir = Path(sys.argv[2]).resolve()
project_dir = Path(sys.argv[3]).resolve()
policy = sys.argv[4]
datasets = sys.argv[5:]

def normalized_key(path):
    return unicodedata.normalize('NFC', str(path)).casefold()

# Git can preserve decomposed Unicode names while TSV text is composed. Build a
# lookup once so both forms point at the same real file on Linux and Windows.
file_lookup = {}
image_root = (project_dir.parent.parent / 'dermnet-output' / 'images').resolve()
if image_root.is_dir():
    for path in image_root.rglob('*'):
        if path.is_file():
            file_lookup[normalized_key(path)] = path

had_missing = False
for dataset in datasets:
    source = source_dir / f'{dataset}.tsv'
    target = runtime_dir / source.name
    frame = pd.read_csv(source, sep='\t')
    required = {'index', 'question', 'image_path'}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise SystemExit(f'{source.name}: missing columns {missing_columns}')
    if frame['index'].duplicated().any():
        raise SystemExit(f'{source.name}: duplicate index values')

    keep = []
    resolved_paths = []
    missing_records = []
    normalized_count = 0
    for row_number, raw in enumerate(frame['image_path'].astype(str), start=2):
        portable = raw.replace('\\', os.sep).replace('/', os.sep)
        candidate = Path(portable)
        if not candidate.is_absolute():
            candidate = (project_dir / candidate).resolve()
        actual = candidate if candidate.is_file() else file_lookup.get(normalized_key(candidate))
        if actual is None:
            keep.append(False)
            resolved_paths.append('')
            missing_records.append(f'row={row_number}\tindex={frame.iloc[row_number - 2]["index"]}\t{raw}')
        else:
            keep.append(True)
            resolved_paths.append(str(actual))
            normalized_count += int(actual != candidate)

    missing_path = runtime_dir / f'{dataset}.missing-images.txt'
    missing_path.write_text('\n'.join(missing_records) + ('\n' if missing_records else ''), encoding='utf-8')
    if missing_records:
        had_missing = True
        if policy == 'fail':
            print(f'{source.name}: {len(missing_records)} missing rows; see {missing_path}', file=sys.stderr)
            continue
    output = frame.loc[keep].copy()
    output['image_path'] = [path for path, is_kept in zip(resolved_paths, keep) if is_kept]
    output.to_csv(target, sep='\t', index=False)
    print(
        f'{source.name}: source={len(frame)}, runtime={len(output)}, '
        f'unicode_fixed={normalized_count}, skipped={len(missing_records)}'
    )

if had_missing and policy == 'fail':
    raise SystemExit('Missing images detected; use MISSING_IMAGE_POLICY=skip to create audited runtime TSV files')
PY
    export LMUData="$RUNTIME_DATA_DIR"
}

acquire_lock() {
    mkdir -p "$STATE_DIR" "$COMPLETED_DIR" "$LOG_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null || die "Another runner is active, or the lock is stale: $LOCK_DIR"
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
}

release_lock() {
    if [[ -d "$LOCK_DIR" ]]; then
        rm -f -- "$LOCK_DIR/pid"
        rmdir -- "$LOCK_DIR" 2>/dev/null || true
    fi
}

on_signal() {
    log 'Interrupted. Checkpoints were kept; run `bash run_phase2.sh resume`.'
    exit 130
}

python_for_model() {
    local model="$1"
    if [[ "$model" == deepseek_vl2* ]]; then
        printf '%s\n' "$PYTHON_DEEPSEEK"
    elif [[ "$model" == Qwen3* ]]; then
        printf '%s\n' "$PYTHON_QWEN"
    else
        printf '%s\n' "$PYTHON_LEGACY"
    fi
}

validate_environment() {
    [[ "$DRY_RUN" == '1' ]] && return
    local model python_exe imports
    for model in "${MODELS[@]}"; do
        python_exe="$(python_for_model "$model")"
        command -v "$python_exe" >/dev/null 2>&1 || die "Python executable not found: $python_exe"
        case "$model" in
            Qwen3*) imports='import pandas, torch, transformers, vllm' ;;
            deepseek_vl2_int*) imports='import pandas, torch, transformers, deepseek_vl2, bitsandbytes' ;;
            deepseek_vl2*) imports='import pandas, torch, transformers, deepseek_vl2' ;;
            LLaVA*) imports='import pandas, torch, transformers, llava' ;;
            *) imports='import pandas, torch, transformers' ;;
        esac
        "$python_exe" -c "$imports" >/dev/null 2>&1 \
            || die "Environment for $model is incomplete: $python_exe"
    done
    PYTHON_EXE="$(python_for_model "${MODELS[0]}")"
}

validate_dataset() {
    local dataset="$1"
    [[ "$DRY_RUN" == '1' ]] && return
    "$PYTHON_EXE" - "$LMUData/$dataset.tsv" <<'PY'
import os
import sys
import pandas as pd

tsv_path = os.path.abspath(sys.argv[1])
frame = pd.read_csv(tsv_path, sep='\t')
required = {'index', 'question', 'image_path'}
missing_columns = sorted(required.difference(frame.columns))
if missing_columns:
    raise SystemExit(f'Dataset is missing columns: {missing_columns}')
if frame['index'].duplicated().any():
    raise SystemExit('Dataset contains duplicate index values')

base = os.path.dirname(tsv_path)
missing = []
for raw in frame['image_path'].dropna().astype(str):
    normalized = raw.replace('\\', os.sep).replace('/', os.sep)
    path = normalized if os.path.isabs(normalized) else os.path.normpath(os.path.join(base, normalized))
    if not os.path.isfile(path):
        missing.append(raw)
print(f'{os.path.basename(tsv_path)}: {len(frame)} rows, {len(missing)} missing image references')
if missing:
    for path in sorted(set(missing))[:10]:
        print(f'  missing: {path}', file=sys.stderr)
    raise SystemExit('Dataset is incomplete; refusing to publish a partial benchmark')
PY
}

find_complete_result() {
    local model="$1" dataset="$2"
    "$PYTHON_EXE" - "$SCRIPT_DIR" "$model" "$dataset" <<'PY'
import json
import os
import sys
from pathlib import Path
import pandas as pd

root, model, dataset = sys.argv[1:]
model_dir = Path(root) / 'outputs' / model
dataset_file = Path(os.environ['LMUData']) / f'{dataset}.tsv'
if not model_dir.is_dir() or not dataset_file.is_file():
    raise SystemExit(1)
expected = set(pd.read_csv(dataset_file, sep='\t', usecols=['index'])['index'].astype(str))
for status_file in sorted(model_dir.glob('*/status.json'), key=lambda p: p.stat().st_mtime, reverse=True):
    try:
        status = json.loads(status_file.read_text(encoding='utf-8'))
        item = status.get('datasets', {}).get(dataset, {})
        raw_result = item.get('prediction_file')
        if item.get('status') != 'done' or not raw_result:
            continue
        result = Path(raw_result)
        if not result.is_absolute():
            result = Path(root) / result
        frame = pd.read_excel(result) if result.suffix == '.xlsx' else pd.read_csv(result, sep='\t')
        if not {'index', 'prediction'}.issubset(frame.columns):
            continue
        predictions = frame[['index', 'prediction']].dropna(subset=['prediction'])
        failed = predictions['prediction'].astype(str).str.contains(
            'Failed to obtain answer', case=False, regex=False).any()
        if expected.issubset(set(predictions['index'].astype(str))) and not failed:
            print(result)
            raise SystemExit(0)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        continue
raise SystemExit(1)
PY
}

run_job() {
    local model="$1" dataset="$2"
    local id marker job_log attempt result rc
    PYTHON_EXE="$(python_for_model "$model")"
    id="$(job_id "$model" "$dataset")"
    marker="$COMPLETED_DIR/$id.ok"
    job_log="$LOG_DIR/$id.log"

    if [[ "$DRY_RUN" != '1' ]] && result="$(find_complete_result "$model" "$dataset")"; then
        printf 'completed_at=%s\nresult=%s\n' "$(timestamp)" "$result" > "$marker"
        log "SKIP $id (verified complete)"
        return 0
    fi

    if [[ "$DRY_RUN" == '1' ]]; then
        printf '%q ' "$PYTHON_EXE" run.py --data "$dataset" --model "$model" \
            --mode infer --verbose --reuse --reuse-aux infer
        printf '\n'
        return 0
    fi

    validate_dataset "$dataset"
    for ((attempt = 1; attempt <= MAX_JOB_RETRIES; attempt++)); do
        log "RUN $id (attempt $attempt/$MAX_JOB_RETRIES)"
        set +e
        "$PYTHON_EXE" run.py --data "$dataset" --model "$model" \
            --mode infer --verbose --reuse --reuse-aux infer 2>&1 | tee -a "$job_log"
        rc=${PIPESTATUS[0]}
        set -e
        if [[ "$rc" -eq 0 ]] && result="$(find_complete_result "$model" "$dataset")"; then
            printf 'completed_at=%s\nresult=%s\n' "$(timestamp)" "$result" > "$marker"
            log "DONE $id -> $result"
            return 0
        fi
        log "FAILED $id (exit=$rc or incomplete output); checkpoint was kept."
    done
    return 1
}

run_all_jobs() {
    local failed=() model dataset
    show_plan
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            if ! run_job "$model" "$dataset"; then
                failed+=("$(job_id "$model" "$dataset")")
            fi
        done
    done
    if (( ${#failed[@]} > 0 )); then
        log "Finished with ${#failed[@]} failed/incomplete job(s): ${failed[*]}"
        log 'Run `bash run_phase2.sh resume` to retry unfinished jobs.'
        return 1
    fi
    if [[ "$DRY_RUN" == '1' ]]; then
        log 'Dry-run plan is complete; no inference was started.'
    else
        log 'All selected bilingual model/dataset jobs are complete.'
    fi
}

show_status() {
    select_models
    local total=$(( ${#MODELS[@]} * ${#DATASETS[@]} )) done_count=0 model dataset marker state
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            marker="$COMPLETED_DIR/$(job_id "$model" "$dataset").ok"
            if [[ -f "$marker" ]]; then state='DONE'; done_count=$((done_count + 1)); else state='PENDING'; fi
            printf '%-7s %-25s %s\n' "$state" "$model" "$dataset"
        done
    done
    printf 'Progress: %d/%d jobs completed\n' "$done_count" "$total"
}

main() {
    local command="${1:-}"
    case "$command" in
        plan) show_plan ;;
        status) show_status ;;
        all|resume)
            select_models
            validate_static_files
            if [[ "$DRY_RUN" != '1' ]]; then
                acquire_lock
                trap release_lock EXIT
                trap on_signal INT TERM HUP
            fi
            validate_environment
            prepare_datasets
            run_all_jobs
            ;;
        full)
            [[ $# -eq 3 ]] || die 'Usage: bash run_phase2.sh full <model_name> <dataset_name>'
            MODELS=("$2")
            DATASETS=("$3")
            validate_static_files
            if [[ "$DRY_RUN" != '1' ]]; then
                acquire_lock
                trap release_lock EXIT
                trap on_signal INT TERM HUP
            fi
            validate_environment
            prepare_datasets
            run_job "$2" "$3"
            ;;
        -h|--help|help|'') usage ;;
        *) usage >&2; die "Unknown command: $command" ;;
    esac
}

main "$@"
