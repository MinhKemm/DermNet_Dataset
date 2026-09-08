"""Exercise the actual Bash entry point without downloading or loading models."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("bash"), "Bash required")
class RunnerTest(unittest.TestCase):
    def run_shell(self, command):
        return subprocess.run(
            ["bash", "-lc", command], cwd=Path(__file__).parents[1],
            capture_output=True, text=True, timeout=30,
        )

    def test_full_plan_has_eight_full_and_eight_patch_jobs(self):
        result = self.run_shell(
            "GPU_COUNT=1 GPU_MAX_VRAM_GB=80 GPU_TOTAL_VRAM_GB=80 "
            "MODEL_PROFILE=full bash run_phase2.sh plan"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = [line for line in result.stdout.splitlines() if '. full ' in line or '. patch ' in line]
        self.assertEqual(16, len(jobs))
        self.assertEqual(8, sum('. full ' in line for line in jobs))
        self.assertEqual(8, sum('. patch ' in line for line in jobs))
        self.assertNotIn('DermNet_Val_4k_en', result.stdout)
        self.assertNotIn('deepseek_vl2_int4 ', result.stdout)

    def test_missing_patch_files_stop_before_inference(self):
        result = self.run_shell(
            "test_dir=$(mktemp -d); "
            "GPU_COUNT=1 GPU_MAX_VRAM_GB=80 GPU_TOTAL_VRAM_GB=80 "
            "DRY_RUN=1 LEGACY_RESULTS_DIR=$test_dir bash run_phase2.sh all; "
            "rc=$?; rmdir -- \"$test_dir\"; exit $rc"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(8, result.stdout.count('MISSING patch input:'))
        self.assertNotIn('python3 run.py', result.stdout)

    def test_small_gpu_keeps_only_legacy_models_that_fit(self):
        result = self.run_shell(
            "GPU_COUNT=1 GPU_MAX_VRAM_GB=12 GPU_TOTAL_VRAM_GB=12 "
            "MODEL_PROFILE=auto bash run_phase2.sh plan"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = [line for line in result.stdout.splitlines() if '. full ' in line or '. patch ' in line]
        self.assertEqual(6, len(jobs))
        self.assertFalse(any('int4' in line or 'int8' in line for line in jobs))


if __name__ == '__main__':
    unittest.main()
