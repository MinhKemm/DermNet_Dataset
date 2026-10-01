#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
ROOT_DIR="$(cd -- "$KIT_DIR/../.." && pwd -P)"
REQ_DIR="$KIT_DIR/requirements/server"
VENDOR_DIR="${DERMNET_VENDOR_DIR:-$ROOT_DIR/vendor}"
ENV_FILE="${SERVER_ENV_FILE:-$KIT_DIR/.phase2-server-env.sh}"
FIX_QWEN_SCRIPT="$SCRIPT_DIR/fix_qwen_vllm_env.sh"

# FlashInfer 0.6.16 requires Python 3.12 for its runtime annotations.
# Keep this stack separate from the legacy Transformers environments.
VLLM_ENV="${VLLM_ENV:-dermnet-vllm-py312}"
DEEPSEEK_ENV="${DEEPSEEK_ENV:-dermnet-deepseek-int8}"
VINTERN_ENV="${VINTERN_ENV:-dermnet-vintern}"
HUATUO_ENV="${HUATUO_ENV:-dermnet-huatuo}"
LEGACY_TORCH_INDEX_URL="${LEGACY_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MODE="${1:-install}"

DEEPSEEK_COMMIT='ef9f91e2b6426536b83294c11742c27be66361b1'
HUATUO_COMMIT='e1a52dcf6c0417f4b6ac1d378b01147280192fca'

log() { printf '[setup] %s\n' "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
case "$MODE" in
    install|prepare-runtime) ;;
    *) die 'Usage: setup_server_envs.sh [install|prepare-runtime]' ;;
esac
command -v conda >/dev/null 2>&1 || die 'conda was not found in PATH.'
command -v git >/dev/null 2>&1 || die 'git was not found in PATH.'
[[ -f "$FIX_QWEN_SCRIPT" ]] || die "Missing vLLM fix script: $FIX_QWEN_SCRIPT"
if [[ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ]]; then
    die 'This is a login-node setup script. Do not run it inside a scheduler/compute job.'
fi

ensure_env() {
    local name="$1" python_version="${2:-3.10}" actual_version
    if ! conda run -n "$name" python -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
        log "Creating Conda environment: $name (Python $python_version)"
        conda create -n "$name" "python=$python_version" pip -y
    else
        actual_version="$(conda run --no-capture-output -n "$name" python -c \
            'import sys; print(".".join(map(str, sys.version_info[:2])))')"
        [[ "$actual_version" == "$python_version" ]] || die \
            "Environment $name uses Python $actual_version; this profile requires Python $python_version."
    fi
    conda run -n "$name" python -m pip install --upgrade pip setuptools wheel packaging
}

require_env() {
    local name="$1"
    conda run -n "$name" python -c 'import sys; print(sys.executable)' >/dev/null 2>&1 \
        || die "Conda environment '$name' was not found. Install its requirement profile first."
}

python_path() {
    conda run --no-capture-output -n "$1" python -c 'import sys; print(sys.executable)' \
        | awk 'NF { line=$0 } END { print line }'
}

validate_vllm_runtime() {
    local python_exe="$1" clean_python actual_version
    actual_version="$("$python_exe" -c \
        'import sys; print(".".join(map(str, sys.version_info[:2])))')"
    [[ "$actual_version" == '3.12' ]] || die \
        "vLLM environment uses Python $actual_version; run scripts/fix_qwen_vllm_env.sh to create a Python 3.12 environment."

    clean_python="$(dirname -- "$python_exe")/dermnet-vllm-clean"
    [[ -x "$clean_python" ]] || die \
        "Missing clean vLLM launcher: $clean_python. Run scripts/fix_qwen_vllm_env.sh first."

    "$clean_python" -m pip check
    "$clean_python" - <<'PY'
import importlib.metadata as metadata
import os
from array import array

assert "LD_LIBRARY_PATH" not in os.environ
array[int]
import flashinfer.comm  # noqa: F401
import vllm.distributed.device_communicators.flashinfer_all_reduce  # noqa: F401

for distribution, wanted in {
    "vllm": "0.28.0",
    "flashinfer-python": "0.6.16.post3",
    "flashinfer-cubin": "0.6.16.post3",
}.items():
    actual = metadata.version(distribution)
    if actual != wanted:
        raise RuntimeError(f"{distribution}=={actual}; expected {wanted}")
PY
}

install_profile() {
    local name="$1" profile="$2"
    log "Installing profile $profile into $name"
    conda run -n "$name" python -m pip install -r "$REQ_DIR/$profile"
    conda run -n "$name" python -m pip install --no-deps -e "$KIT_DIR"
}

install_legacy_torch() {
    local name="$1"
    conda run -n "$name" python -m pip install \
        torch==2.8.0 torchvision==0.23.0 --index-url "$LEGACY_TORCH_INDEX_URL"
}

ensure_repo() {
    local url="$1" path="$2" commit="$3"
    if [[ ! -d "$path/.git" ]]; then
        git clone "$url" "$path"
        git -C "$path" checkout --detach "$commit"
    else
        local actual
        actual="$(git -C "$path" rev-parse HEAD)"
        [[ "$actual" == "$commit" ]] || die "$path is at $actual; expected pinned commit $commit. Move it aside or set DERMNET_VENDOR_DIR."
    fi
}

mkdir -p "$VENDOR_DIR"
if [[ "$MODE" == install ]]; then
    ensure_env "$VLLM_ENV" 3.12

    ensure_env "$DEEPSEEK_ENV"
    install_legacy_torch "$DEEPSEEK_ENV"
    install_profile "$DEEPSEEK_ENV" deepseek-int8-blackwell.txt

    ensure_env "$VINTERN_ENV"
    install_legacy_torch "$VINTERN_ENV"
    install_profile "$VINTERN_ENV" vintern-blackwell.txt

    ensure_env "$HUATUO_ENV"
    install_legacy_torch "$HUATUO_ENV"
    install_profile "$HUATUO_ENV" huatuo-blackwell.txt
else
    log 'Skipping package installation; preparing the four existing environments.'
    require_env "$VLLM_ENV"
    require_env "$DEEPSEEK_ENV"
    require_env "$VINTERN_ENV"
    require_env "$HUATUO_ENV"
fi

ensure_repo https://github.com/deepseek-ai/DeepSeek-VL2.git "$VENDOR_DIR/DeepSeek-VL2" "$DEEPSEEK_COMMIT"
ensure_repo https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git "$VENDOR_DIR/HuatuoGPT-Vision" "$HUATUO_COMMIT"
# Register the source tree without its upstream torch==2.0.1 package metadata;
# that wheel cannot target Blackwell. All runtime dependencies are explicit in
# our profile and are still verified by pip check and the compute-job preflight.
conda run -n "$DEEPSEEK_ENV" python -c \
    'import site,sys; from pathlib import Path; Path(site.getsitepackages()[0], "dermnet_deepseek_vl2.pth").write_text(sys.argv[1] + "\n")' \
    "$VENDOR_DIR/DeepSeek-VL2"
conda run -n "$HUATUO_ENV" python "$SCRIPT_DIR/patch_vendor_sources.py" \
    --deepseek-dir "$VENDOR_DIR/DeepSeek-VL2" \
    --huatuo-dir "$VENDOR_DIR/HuatuoGPT-Vision"

VLLM_PYTHON="$(python_path "$VLLM_ENV")"
if [[ "$MODE" == install ]]; then
    log 'Applying the canonical Qwen/vLLM fix and checking all legacy environments'
    SERVER_ENV_FILE="$ENV_FILE" \
    QWEN_VLLM_ENV="$VLLM_ENV" \
    DEEPSEEK_ENV="$DEEPSEEK_ENV" \
    VINTERN_ENV="$VINTERN_ENV" \
    HUATUO_ENV="$HUATUO_ENV" \
    CHECK_ALL_ENVS=1 \
        bash "$FIX_QWEN_SCRIPT"
else
    log 'Validating the preinstalled vLLM runtime and clean launcher'
    validate_vllm_runtime "$VLLM_PYTHON"
fi

conda run -n "$VLLM_ENV" python -m pip check
conda run -n "$DEEPSEEK_ENV" python -m pip check
conda run -n "$VINTERN_ENV" python -m pip check
conda run -n "$HUATUO_ENV" python -m pip check

DEEPSEEK_PYTHON="$(python_path "$DEEPSEEK_ENV")"
VINTERN_PYTHON="$(python_path "$VINTERN_ENV")"
HUATUO_PYTHON="$(python_path "$HUATUO_ENV")"
CLEAN_VLLM_PYTHON="$(dirname -- "$VLLM_PYTHON")/dermnet-vllm-clean"
[[ -x "$CLEAN_VLLM_PYTHON" ]] || die "Missing clean vLLM launcher: $CLEAN_VLLM_PYTHON"

write_export() {
    printf 'export %s=%q\n' "$1" "$2" >> "$ENV_FILE"
}

printf '# Generated by scripts/setup_server_envs.sh\n' > "$ENV_FILE"
write_export PYTHON_BIN "$CLEAN_VLLM_PYTHON"
write_export PYTHON_QWEN "$CLEAN_VLLM_PYTHON"
write_export PYTHON_DEEPSEEK_VLLM "$CLEAN_VLLM_PYTHON"
write_export PYTHON_DEEPSEEK "$DEEPSEEK_PYTHON"
write_export PYTHON_VINTERN "$VINTERN_PYTHON"
write_export PYTHON_HUATUO "$HUATUO_PYTHON"
write_export HUATUO_SOURCE_DIR "$VENDOR_DIR/HuatuoGPT-Vision"

log "Saved runtime mapping: $ENV_FILE"
if [[ "$MODE" == install ]]; then
    log 'Environment installation is complete on the login node.'
    log 'Submit a compute job; it will run doctor/CUDA/memory checks before inference.'
else
    log 'Runtime preparation is complete on the login node. CUDA will be checked inside each compute job.'
fi
