#!/usr/bin/env python3
"""Apply the pinned Blackwell-safe inference patches to vendor source trees."""

from __future__ import annotations

import argparse
from pathlib import Path


DEEPSEEK_OLD = '''    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from xformers.ops import memory_efficient_attention

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if not self.qk_norm:
            if self.head_dim % 32 == 0 and is_flash_attn_2_available():
                # flashattn must have head_dim as a multiple of 32
                x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop.p if self.training else 0.,
                                              deterministic=self.deterministic)
            else:
                q, k, v = qkv.unbind(2)
                x = memory_efficient_attention(q, k, v, p=self.attn_drop.p if self.training else 0.)
            x = x.reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False):
                # 用上下文的方式强行使用fa
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
'''

DEEPSEEK_NEW = '''    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if not self.qk_norm:
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

        # Let PyTorch select an attention kernel supported by the installed
        # wheel/GPU. This avoids the legacy source forcing xFormers or FA2,
        # whose prebuilt kernels do not reliably cover Blackwell SM 12.0.
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
'''

HUATUO_OLD = '''        config._attn_implementation = "flash_attention_2"
        config._flash_attn_2_enabled = True
'''

HUATUO_NEW = '''        # Use the portable Transformers attention path. The upstream code
        # forces FlashAttention 2, whose CUDA extension is not reliable on
        # Blackwell SM 12.0 with this legacy Transformers generation.
        config._attn_implementation = "eager"
        config._flash_attn_2_enabled = False
'''


def replace_exact(path: Path, old: str, new: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"{label}: missing source file: {path}")
    text = path.read_text(encoding="utf-8")
    if old in text:
        if text.count(old) != 1:
            raise RuntimeError(f"{label}: expected exactly one upstream block in {path}")
        path.write_text(text.replace(old, new), encoding="utf-8")
        return "patched"
    if new in text:
        return "already patched"
    raise RuntimeError(
        f"{label}: pinned source no longer matches the reviewed block in {path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-dir", type=Path, required=True)
    parser.add_argument("--huatuo-dir", type=Path, required=True)
    args = parser.parse_args()

    deepseek_file = args.deepseek_dir / "deepseek_vl2/models/siglip_vit.py"
    huatuo_file = args.huatuo_dir / "llava/model/language_model/llava_llama.py"
    print("DeepSeek vision attention:", replace_exact(
        deepseek_file, DEEPSEEK_OLD, DEEPSEEK_NEW, "DeepSeek-VL2"
    ))
    print("Huatuo language attention:", replace_exact(
        huatuo_file, HUATUO_OLD, HUATUO_NEW, "HuatuoGPT-Vision"
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
