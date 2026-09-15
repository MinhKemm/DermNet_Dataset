#!/usr/bin/env bash

set -Eeuo pipefail

# Build a clean, reproducible vLLM environment for the Qwen/DeepSeek-vLLM
# jobs on the Blackwell server. The generated launcher also removes the
# problematic LD_LIBRARY_PATH before starting vLLM. Run this once on the
# server setup/login node; do not install packages inside every scheduler job.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
REQ_FILE="$KIT_DIR/requirements/server/vllm-blackwell.txt"
ENV_FILE="${SERVER_ENV_FILE:-$KIT_DIR/.phase2-server-env.sh}"

ENV_NAME="${QWEN_VLLM_ENV:-dermnet-vllm-py312}"
DEEPSEEK_ENV="${DEEPSEEK_ENV:-dermnet-deepseek-int8}"
VINTERN_ENV="${VINTERN_ENV:-dermnet-vintern}"
HUATUO_ENV="${HUATUO_ENV:-dermnet-huatuo}"
CHECK_ALL_ENVS="${CHECK_ALL_ENVS:-1}"
CHECK_GPU_ENVS="${CHECK_GPU_ENVS:-0}"
FLASHINFER_INDEX="https://flashinfer.ai/whl/"
FLASHINFER_PYTHON="flashinfer-python==0.6.16.post3"
FLASHINFER_CUBIN="flashinfer-cubin==0.6.16.post3"

log() { printf '[qwen-fix] %s\n' "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || die \
    'conda was not found in PATH. Load Miniconda/Anaconda before running this script.'
[[ -f "$REQ_FILE" ]] || die "Missing requirements file: $REQ_FILE"

CONDA_BASE="$(conda info --base)"
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || die \
    "Cannot find conda shell integration under $CONDA_BASE."
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

env_python_version="$(conda run -n "$ENV_NAME" python -c \
    'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || true)"

if [[ -z "$env_python_version" ]]; then
    log "Creating Conda environment $ENV_NAME with Python 3.12"
    conda create -n "$ENV_NAME" python=3.12 pip -y
elif [[ "$env_python_version" != '3.12' ]]; then
    die "Environment $ENV_NAME already uses Python $env_python_version; choose another QWEN_VLLM_ENV instead of modifying it."
else
    log "Reusing existing Python 3.12 environment $ENV_NAME"
fi

run_python() {
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
}

log 'Updating packaging tools'
run_python -m pip install --upgrade pip setuptools wheel packaging

log 'Installing the pinned VLMEvalKit server profile'
run_python -m pip install -r "$REQ_FILE"

log "Installing $FLASHINFER_PYTHON and $FLASHINFER_CUBIN"
run_python -m pip install --no-cache-dir \
    --extra-index-url "$FLASHINFER_INDEX" \
    "$FLASHINFER_PYTHON" "$FLASHINFER_CUBIN"

log 'Registering VLMEvalKit without re-resolving its dependencies'
run_python -m pip install --no-deps -e "$KIT_DIR"

VLLM_PYTHON="$(conda run --no-capture-output -n "$ENV_NAME" python -c \
    'import sys; print(sys.executable)')"
VLLM_BIN_DIR="$(dirname -- "$VLLM_PYTHON")"
CLEAN_PYTHON="$VLLM_BIN_DIR/dermnet-vllm-clean"

# vLLM/NVML on this cluster needs LD_LIBRARY_PATH removed. Keep the real
# interpreter untouched and route only vLLM-backed models through this wrapper.
printf -v quoted_python '%q' "$VLLM_PYTHON"
printf '#!/usr/bin/env bash\nexec env -u LD_LIBRARY_PATH %s "$@"\n' \
    "$quoted_python" > "$CLEAN_PYTHON"
chmod 755 "$CLEAN_PYTHON"

log 'Checking package consistency and FlashInfer/vLLM imports'
"$CLEAN_PYTHON" -m pip check
"$CLEAN_PYTHON" - <<'PY'
import importlib.metadata as metadata
import os
import sys
from array import array

assert sys.version_info[:2] == (3, 12), sys.version
assert "LD_LIBRARY_PATH" not in os.environ
array[int]
import flashinfer.comm  # noqa: F401
import vllm.distributed.device_communicators.flashinfer_all_reduce  # noqa: F401

expected = {
    "vllm": "0.28.0",
    "flashinfer-python": "0.6.16.post3",
    "flashinfer-cubin": "0.6.16.post3",
}
for distribution, wanted in expected.items():
    actual = metadata.version(distribution)
    if actual != wanted:
        raise RuntimeError(f"{distribution}=={actual}; expected {wanted}")

print(f"Python: {sys.version.split()[0]}")
for distribution in expected:
    print(f"{distribution}: {metadata.version(distribution)}")
print("FlashInfer import: OK")
print("vLLM FlashInfer communicator import: OK")
print("LD_LIBRARY_PATH: unset for vLLM launcher")
PY

# The normal runner reads this ignored/generated file. Preserve all mappings
# for the other model profiles and update only the vLLM-backed Qwen jobs.
update_runtime_mapping() {
    local tmp
    printf -v quoted_python '%q' "$CLEAN_PYTHON"

    if [[ -f "$ENV_FILE" ]]; then
        cp -- "$ENV_FILE" "$ENV_FILE.bak"
        tmp="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
        awk -v value="$quoted_python" '
            /^export PYTHON_QWEN=/ {
                print "export PYTHON_QWEN=" value
                seen_qwen=1
                next
            }
            /^export PYTHON_DEEPSEEK_VLLM=/ {
                print "export PYTHON_DEEPSEEK_VLLM=" value
                seen_deepseek_vllm=1
                next
            }
            { print }
            END {
                if (!seen_qwen) print "export PYTHON_QWEN=" value
                if (!seen_deepseek_vllm) print "export PYTHON_DEEPSEEK_VLLM=" value
            }
        ' "$ENV_FILE" > "$tmp"
        mv -- "$tmp" "$ENV_FILE"
    else
        tmp="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
        {
            printf '# Generated by scripts/fix_qwen_vllm_env.sh\n'
            printf 'export PYTHON_QWEN=%s\n' "$quoted_python"
            printf 'export PYTHON_DEEPSEEK_VLLM=%s\n' "$quoted_python"
        } > "$tmp"
        mv -- "$tmp" "$ENV_FILE"
    fi
}

update_runtime_mapping

check_existing_env() {
    local name="$1" role="$2"
    local env_python

    env_python="$(conda run -n "$name" python -c \
        'import sys; print(sys.executable)' 2>/dev/null || true)"
    if [[ -z "$env_python" ]]; then
        die "Environment $name is missing. Set CHECK_ALL_ENVS=0 for a Qwen-only setup."
    fi

    log "Checking package consistency: $name"
    conda run --no-capture-output -n "$name" python -m pip check

    if [[ "$CHECK_GPU_ENVS" == '1' ]]; then
        log "Running GPU profile check: $name ($role)"
        if [[ "$role" == 'huatuo' ]]; then
            HUATUO_SOURCE_DIR="${HUATUO_SOURCE_DIR:-$KIT_DIR/../../vendor/HuatuoGPT-Vision}"
            env HUATUO_SOURCE_DIR="$HUATUO_SOURCE_DIR" \
                conda run --no-capture-output -n "$name" python \
                "$KIT_DIR/scripts/check_server_env.py" "$role"
        else
            conda run --no-capture-output -n "$name" python \
                "$KIT_DIR/scripts/check_server_env.py" "$role"
        fi
    fi
}

if [[ "$CHECK_ALL_ENVS" == '1' ]]; then
    # The legacy model families intentionally keep their own Python/Torch
    # stacks. Check them, but do not force the Python 3.12 vLLM environment on
    # them and do not apply the vLLM-only LD_LIBRARY_PATH workaround to them.
    check_existing_env "$DEEPSEEK_ENV" deepseek-int8
    check_existing_env "$VINTERN_ENV" vintern
    check_existing_env "$HUATUO_ENV" huatuo
fi

log "Runtime mapping updated: $ENV_FILE"
log "Qwen launcher: $CLEAN_PYTHON"
log "Qwen interpreter: $VLLM_PYTHON"
if [[ "$CHECK_GPU_ENVS" != '1' ]]; then
    log 'GPU-specific checks were skipped; set CHECK_GPU_ENVS=1 inside an allocated GPU job.'
fi
log 'Environment fix passed. Run one Qwen smoke test before the full vLLM group.'
