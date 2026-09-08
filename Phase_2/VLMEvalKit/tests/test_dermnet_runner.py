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

    def test_server_plan_has_twelve_full_and_four_patch_jobs(self):
        result = self.run_shell(
            "GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 "
            "MODEL_PROFILE=auto bash run_phase2.sh plan"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = [line for line in result.stdout.splitlines() if '. full ' in line or '. patch ' in line]
        self.assertEqual(16, len(jobs))
        self.assertEqual(12, sum('. full ' in line for line in jobs))
        self.assertEqual(2, sum('HuatuoGPT-Vision-' in line for line in jobs))
        self.assertEqual(4, sum('. patch ' in line for line in jobs))
        self.assertNotIn('_EN', result.stdout)
        self.assertNotIn('Gemma', result.stdout)
        self.assertNotIn('LLaVA-med', result.stdout)
        small = [line for line in jobs if 'deepseek_vl2_small' in line]
        self.assertEqual(2, len(small))
        self.assertTrue(all('. full ' in line for line in small))
        int8 = [line for line in jobs if 'deepseek_vl2_int8' in line]
        self.assertEqual(2, len(int8))
        self.assertTrue(all('. patch ' in line for line in int8))
        tiny = [line for line in jobs if 'deepseek_vl2_tiny' in line]
        self.assertEqual(2, len(tiny))
        self.assertTrue(all('. patch ' in line for line in tiny))
        self.assertFalse(any('. patch ' in line and 'Vintern' in line for line in jobs))
        self.assertEqual(4, sum('Vintern-' in line for line in jobs))
        self.assertNotIn('deepseek_vl2_int4 ', result.stdout)

    def test_missing_patch_files_stop_before_inference(self):
        result = self.run_shell(
            "GPU_COUNT=1 GPU_MAX_VRAM_GB=80 GPU_TOTAL_VRAM_GB=80 "
            "DRY_RUN=1 LEGACY_RESULTS_DIR=/__dermnet_nonexistent_test_inputs__ bash run_phase2.sh all"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(4, result.stdout.count('MISSING patch input:'))
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

    def test_all_dry_run_with_real_manifest_and_sources(self):
        result = self.run_shell(
            'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
            'DRY_RUN=1 bash run_phase2.sh all'
        )
        self.assertEqual(0, result.returncode, result.stderr + result.stdout[-1000:])
        self.assertIn('Dry-run plan is complete', result.stdout)
        self.assertEqual(16, result.stdout.count(' run.py --data '))
        vintern_commands = [line for line in result.stdout.splitlines() if ' run.py --data ' in line and '--model Vintern-' in line]
        self.assertEqual(4, len(vintern_commands))
        self.assertTrue(all('vintern-full-rerun-20260908' in line for line in vintern_commands))

    def test_vintern_can_use_separate_python(self):
        result = self.run_shell(
            'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
            'MODEL_PROFILE=full DRY_RUN=1 PYTHON_VINTERN=/env/vintern/bin/python '
            'PYTHON_LEGACY=/env/llava/bin/python bash run_phase2.sh all'
        )
        self.assertEqual(0, result.returncode, result.stderr)
        commands = [line for line in result.stdout.splitlines() if ' run.py --data ' in line]
        self.assertTrue(all(line.startswith('/env/vintern/bin/python ') for line in commands if '--model Vintern-' in line))
        self.assertFalse(any('--model LLaVA-med-' in line for line in commands))

    def test_default_plan_excludes_llava_and_routes_deepseek(self):
        result = self.run_shell(
            'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
            'DRY_RUN=1 PYTHON_DEEPSEEK_VLLM=/env/vllm/bin/python '
            'PYTHON_DEEPSEEK=/env/int8/bin/python bash run_phase2.sh all'
        )
        self.assertEqual(0, result.returncode, result.stderr)
        commands = [line for line in result.stdout.splitlines() if ' run.py --data ' in line]
        self.assertEqual(16, len(commands))
        self.assertFalse(any('--model LLaVA-med-' in line for line in commands))
        self.assertTrue(all(line.startswith('/env/vllm/bin/python ') for line in commands if '--model deepseek_vl2_tiny ' in line or '--model deepseek_vl2_small ' in line))
        self.assertTrue(all(line.startswith('/env/int8/bin/python ') for line in commands if '--model deepseek_vl2_int8 ' in line))


if __name__ == '__main__':
    unittest.main()
