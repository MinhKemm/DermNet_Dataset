#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
ROOT_DIR="$(cd -- "$KIT_DIR/../.." && pwd -P)"
REQ_DIR="$KIT_DIR/requirements/server"
VENDOR_DIR="${DERMNET_VENDOR_DIR:-$ROOT_DIR/vendor}"
ENV_FILE="${SERVER_ENV_FILE:-$KIT_DIR/.phase2-server-env.sh}"

VLLM_ENV="${VLLM_ENV:-dermnet-vllm}"
DEEPSEEK_ENV="${DEEPSEEK_ENV:-dermnet-deepseek-int8}"
VINTERN_ENV="${VINTERN_ENV:-dermnet-vintern}"
HUATUO_ENV="${HUATUO_ENV:-dermnet-huatuo}"
LEGACY_TORCH_INDEX_URL="${LEGACY_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
LEGACY_TORCH="${LEGACY_TORCH:-torch==2.8.0}"
LEGACY_TORCHVISION="${LEGACY_TORCHVISION:-torchvision==0.23.0}"

DEEPSEEK_COMMIT='ef9f91e2b6426536b83294c11742c27be66361b1'
HUATUO_COMMIT='e1a52dcf6c0417f4b6ac1d378b01147280192fca'

log() { printf '[setup] %s\n' "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || die 'conda was not found in PATH.'
command -v git >/dev/null 2>&1 || die 'git was not found in PATH.'
command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi was not found; run this on the NVIDIA compute server.'
command -v nvcc >/dev/null 2>&1 || die 'nvcc was not found; load the CUDA toolkit module required to build Huatuo FlashAttention.'

ensure_env() {
    local name="$1"
    if ! conda run -n "$name" python -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
        log "Creating Conda environment: $name"
        conda create -n "$name" python=3.10 pip -y
    fi
    conda run -n "$name" python -m pip install --upgrade pip setuptools wheel packaging
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
        "$LEGACY_TORCH" "$LEGACY_TORCHVISION" --index-url "$LEGACY_TORCH_INDEX_URL"
}

check_nvcc_matches_torch() {
    local name="$1" torch_cuda nvcc_cuda
    torch_cuda="$(conda run -n "$name" python -c 'import torch; print(torch.version.cuda or "")' | tail -n 1)"
    nvcc_cuda="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)"
    [[ -n "$torch_cuda" && -n "$nvcc_cuda" ]] || die 'Could not determine torch CUDA and nvcc CUDA versions.'
    [[ "$torch_cuda" == "$nvcc_cuda" ]] || die \
        "Huatuo FlashAttention build needs matching CUDA: torch=$torch_cuda, nvcc=$nvcc_cuda. Load CUDA $torch_cuda toolkit, then rerun setup."
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
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

ensure_env "$VLLM_ENV"
install_profile "$VLLM_ENV" vllm-blackwell.txt
conda run -n "$VLLM_ENV" python -m pip check

ensure_env "$DEEPSEEK_ENV"
install_legacy_torch "$DEEPSEEK_ENV"
install_profile "$DEEPSEEK_ENV" deepseek-int8-blackwell.txt
ensure_repo https://github.com/deepseek-ai/DeepSeek-VL2.git "$VENDOR_DIR/DeepSeek-VL2" "$DEEPSEEK_COMMIT"
# Register the source tree without its upstream torch==2.0.1 package metadata;
# that wheel cannot target Blackwell. All runtime dependencies are explicit in
# our profile and are still verified by pip check + doctor.
conda run -n "$DEEPSEEK_ENV" python -c \
    'import site,sys; from pathlib import Path; Path(site.getsitepackages()[0], "dermnet_deepseek_vl2.pth").write_text(sys.argv[1] + "\n")' \
    "$VENDOR_DIR/DeepSeek-VL2"
conda run -n "$DEEPSEEK_ENV" python -m pip check

ensure_env "$VINTERN_ENV"
install_legacy_torch "$VINTERN_ENV"
install_profile "$VINTERN_ENV" vintern-blackwell.txt
conda run -n "$VINTERN_ENV" python -m pip check

ensure_env "$HUATUO_ENV"
install_legacy_torch "$HUATUO_ENV"
install_profile "$HUATUO_ENV" huatuo-blackwell.txt
ensure_repo https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git "$VENDOR_DIR/HuatuoGPT-Vision" "$HUATUO_COMMIT"
# Huatuo's model code explicitly selects FlashAttention 2. It must be built
# against the torch/CUDA already installed in this environment.
check_nvcc_matches_torch "$HUATUO_ENV"
conda run -n "$HUATUO_ENV" python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
conda run -n "$HUATUO_ENV" python -m pip check

python_path() {
    conda run -n "$1" python -c 'import sys; print(sys.executable)' | tail -n 1
}
VLLM_PYTHON="$(python_path "$VLLM_ENV")"
DEEPSEEK_PYTHON="$(python_path "$DEEPSEEK_ENV")"
VINTERN_PYTHON="$(python_path "$VINTERN_ENV")"
HUATUO_PYTHON="$(python_path "$HUATUO_ENV")"

cat > "$ENV_FILE" <<EOF
# Generated by scripts/setup_server_envs.sh
export PYTHON_BIN='$VLLM_PYTHON'
export PYTHON_QWEN='$VLLM_PYTHON'
export PYTHON_DEEPSEEK_VLLM='$VLLM_PYTHON'
export PYTHON_DEEPSEEK='$DEEPSEEK_PYTHON'
export PYTHON_VINTERN='$VINTERN_PYTHON'
export PYTHON_HUATUO='$HUATUO_PYTHON'
export HUATUO_SOURCE_DIR='$VENDOR_DIR/HuatuoGPT-Vision'
EOF

log "Saved runtime mapping: $ENV_FILE"
log 'Run: bash run_phase2.sh doctor'
