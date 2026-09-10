"""Verify deterministic and idempotent patches for pinned legacy sources."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "patch_vendor_sources.py"
SPEC = importlib.util.spec_from_file_location("patch_vendor_sources", SCRIPT)
PATCHER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PATCHER)


class VendorPatchTest(unittest.TestCase):
    def test_deepseek_patch_is_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "siglip_vit.py"
            target.write_text("prefix\n" + PATCHER.DEEPSEEK_OLD + "suffix\n", encoding="utf-8")
            self.assertEqual(
                "patched",
                PATCHER.replace_exact(
                    target, PATCHER.DEEPSEEK_OLD, PATCHER.DEEPSEEK_NEW, "DeepSeek"
                ),
            )
            self.assertEqual(
                "already patched",
                PATCHER.replace_exact(
                    target, PATCHER.DEEPSEEK_OLD, PATCHER.DEEPSEEK_NEW, "DeepSeek"
                ),
            )
            result = target.read_text(encoding="utf-8")
            self.assertIn("scaled_dot_product_attention", result)
            self.assertNotIn("memory_efficient_attention", result)
            self.assertNotIn("enable_math=False", result)

    def test_huatuo_patch_disables_forced_flash_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "llava_llama.py"
            target.write_text(PATCHER.HUATUO_OLD, encoding="utf-8")
            PATCHER.replace_exact(
                target, PATCHER.HUATUO_OLD, PATCHER.HUATUO_NEW, "Huatuo"
            )
            result = target.read_text(encoding="utf-8")
            self.assertIn('config._attn_implementation = "eager"', result)
            self.assertIn("config._flash_attn_2_enabled = False", result)

    def test_unknown_upstream_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "unknown.py"
            target.write_text("changed upstream", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                PATCHER.replace_exact(target, "old", "new", "vendor")


if __name__ == "__main__":
    unittest.main()
