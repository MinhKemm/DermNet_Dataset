#!/usr/bin/env python3
"""Fail-fast checks for the four server inference environments."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def require_package(name: str, minimum: str | None = None, maximum: str | None = None) -> str:
    try:
        installed = version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"missing package: {name}") from exc
    parsed = Version(installed)
    if minimum and parsed < Version(minimum):
        raise RuntimeError(f"{name}=={installed}, need >= {minimum}")
    if maximum and parsed >= Version(maximum):
        raise RuntimeError(f"{name}=={installed}, need < {maximum}")
    return installed


def require_import(*names: str) -> None:
    for name in names:
        importlib.import_module(name)


def check_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    capability = torch.cuda.get_device_capability(0)
    # RTX PRO 6000 Blackwell is SM 12.0. A wheel without its kernels commonly
    # imports successfully but fails only when inference starts.
    if capability >= (12, 0):
        arch_list = torch.cuda.get_arch_list()
        if not any(arch in arch_list for arch in ("sm_120", "compute_120")):
            raise RuntimeError(
                f"torch {torch.__version__} sees SM {capability[0]}.{capability[1]} "
                f"but its wheel has no sm_120/compute_120 kernels: {arch_list}"
            )
    return f"torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_server_env.py vllm|deepseek-int8|vintern|huatuo")
    role = sys.argv[1]
    require_import("pandas", "openpyxl", "PIL")
    details = [check_cuda(), f"transformers={require_package('transformers')}"]

    if role == "vllm":
        details.append(f"vllm={require_package('vllm', '0.28.0', '0.29.0')}")
        require_package("torch", "2.13.0", "2.14.0")
        require_package("torchvision", "0.28.0", "0.29.0")
        require_package("transformers", "5.17.0", "5.18.0")
        require_import("vllm", "qwen_vl_utils")
    elif role == "deepseek-int8":
        require_package("torch", "2.8.0", "2.9.0")
        require_package("torchvision", "0.23.0", "0.24.0")
        require_package("transformers", "4.38.2", "4.39")
        details.append(f"bitsandbytes={require_package('bitsandbytes', '0.49.0')}")
        require_import(
            "deepseek_vl2", "bitsandbytes", "accelerate", "timm",
            "sentencepiece",
        )
        from deepseek_vl2.models.siglip_vit import Attention
        attention_source = inspect.getsource(Attention.forward)
        if "scaled_dot_product_attention" not in attention_source or "xformers" in attention_source:
            raise RuntimeError("DeepSeek Blackwell attention patch is missing")
    elif role == "vintern":
        require_package("torch", "2.8.0", "2.9.0")
        require_package("torchvision", "0.23.0", "0.24.0")
        require_package("transformers", "4.42.3", "5")
        require_import("torchvision", "timm", "sentencepiece", "einops")
    elif role == "huatuo":
        require_package("torch", "2.8.0", "2.9.0")
        require_package("torchvision", "0.23.0", "0.24.0")
        require_package("transformers", "4.37.2", "4.38.0")
        require_import("torchvision", "timm", "peft", "einops", "einops_exts")
        source = os.environ.get("HUATUO_SOURCE_DIR", "")
        if not os.path.isfile(os.path.join(source, "cli.py")):
            raise RuntimeError("HUATUO_SOURCE_DIR/cli.py is missing")
        sys.path.insert(0, source)
        require_import("cli")
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        attention_source = inspect.getsource(LlavaLlamaForCausalLM.__init__)
        if 'config._attn_implementation = "eager"' not in attention_source:
            raise RuntimeError("Huatuo Blackwell attention patch is missing")
    else:
        raise RuntimeError(f"unknown environment profile: {role}")

    print(f"OK {role}: " + ", ".join(details))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ENV ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
