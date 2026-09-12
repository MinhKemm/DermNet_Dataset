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

    def test_manifest_parser_accepts_crlf_checkout(self):
        root = Path(__file__).parents[1]
        manifest = root / 'scripts' / 'dermnet_jobs.txt'
        original = manifest.read_bytes()
        try:
            manifest.write_bytes(original.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
            result = self.run_shell(
                'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
                'DRY_RUN=1 bash run_phase2.sh all'
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(16, result.stdout.count(' run.py --data '))
            self.assertNotIn('MISSING patch input:', result.stdout)
        finally:
            manifest.write_bytes(original)

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

    def test_runtime_groups_partition_the_complete_plan(self):
        expected = {
            'vllm': {
                'Qwen3.5-35B-A3B', 'Qwen3-VL-8B-Instruct',
                'deepseek_vl2_small', 'deepseek_vl2_tiny',
            },
            'deepseek-int8': {'deepseek_vl2_int8'},
            'vintern': {'Vintern-1B-v2', 'Vintern-3B-beta'},
            'huatuo': {'HuatuoGPT-Vision-34B'},
        }
        commands_by_group = {}
        for group, models in expected.items():
            result = self.run_shell(
                'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
                f'DRY_RUN=1 bash run_phase2.sh run-group {group}'
            )
            self.assertEqual(0, result.returncode, result.stderr)
            commands = [
                line for line in result.stdout.splitlines()
                if ' run.py --data ' in line
            ]
            commands_by_group[group] = commands
            self.assertTrue(commands)
            self.assertNotIn('setup_server_envs.sh', result.stdout)
            for line in commands:
                self.assertTrue(any(f'--model {model} ' in line for model in models), line)

        all_commands = [line for commands in commands_by_group.values() for line in commands]
        self.assertEqual(16, len(all_commands))
        self.assertEqual(16, len(set(all_commands)))
        self.assertEqual(8, len(commands_by_group['vllm']))
        self.assertEqual(2, len(commands_by_group['deepseek-int8']))
        self.assertEqual(4, len(commands_by_group['vintern']))
        self.assertEqual(2, len(commands_by_group['huatuo']))

    def test_runtime_group_rejects_unknown_name(self):
        result = self.run_shell(
            'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
            'DRY_RUN=1 bash run_phase2.sh run-group unknown'
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn('Runtime group must be one of', result.stdout + result.stderr)

    def test_two_gpu_groups_parallelize_non_vllm_jobs_one_per_gpu(self):
        for group, expected_jobs in {
            'deepseek-int8': 2,
            'vintern': 4,
            'huatuo': 2,
        }.items():
            result = self.run_shell(
                'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
                f'DRY_RUN=1 bash run_phase2.sh run-group {group}'
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn('Two-GPU mode: 2 parallel workers', result.stdout)
            assignments = [
                line for line in result.stdout.splitlines()
                if 'ASSIGN GPU ' in line
            ]
            self.assertEqual(expected_jobs, len(assignments))
            self.assertTrue(any('ASSIGN GPU 0 ' in line for line in assignments))
            self.assertTrue(any('ASSIGN GPU 1 ' in line for line in assignments))

        vllm = self.run_shell(
            'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
            'DRY_RUN=1 bash run_phase2.sh run-group vllm'
        )
        self.assertEqual(0, vllm.returncode, vllm.stderr)
        self.assertIn('Qwen uses both GPUs; DeepSeek uses two parallel workers', vllm.stdout)
        assignments = [line for line in vllm.stdout.splitlines() if 'ASSIGN GPU ' in line]
        self.assertEqual(4, len(assignments))
        self.assertTrue(all('deepseek_vl2_' in line for line in assignments))
        self.assertTrue(any('ASSIGN GPU 0 ' in line for line in assignments))
        self.assertTrue(any('ASSIGN GPU 1 ' in line for line in assignments))

    def test_parallel_full_jobs_use_distinct_work_directories(self):
        result = self.run_shell(
            'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
            'DRY_RUN=1 bash run_phase2.sh run-group huatuo'
        )
        self.assertEqual(0, result.returncode, result.stderr)
        commands = [line for line in result.stdout.splitlines() if ' run.py --data ' in line]
        self.assertEqual(2, len(commands))
        self.assertTrue(all('/two-gpu-jobs/' in line for line in commands))
        self.assertNotEqual(
            commands[0].split(' --work-dir ', 1)[1].split(' --mode ', 1)[0],
            commands[1].split(' --work-dir ', 1)[1].split(' --mode ', 1)[0],
        )

    def test_run_group_workers_one_disables_parallel_workers(self):
        result = self.run_shell(
            'GPU_COUNT=2 GPU_MAX_VRAM_GB=96 GPU_TOTAL_VRAM_GB=192 '
            'RUN_GROUP_WORKERS=1 DRY_RUN=1 bash run_phase2.sh run-group vllm'
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(8, result.stdout.count(' run.py --data '))
        self.assertNotIn('ASSIGN GPU ', result.stdout)

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

    def test_help_documents_setup_doctor_and_current_job_count(self):
        result = self.run_shell('bash run_phase2.sh --help')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('bash run_phase2.sh server', result.stdout)
        self.assertIn('bash run_phase2.sh setup', result.stdout)
        self.assertIn('bash run_phase2.sh prepare-runtime', result.stdout)
        self.assertIn('connects four preinstalled Conda environments without using a GPU', result.stdout)
        self.assertIn('bash run_phase2.sh doctor', result.stdout)
        self.assertIn('bash run_phase2.sh run-group <vllm|deepseek-int8|vintern|huatuo>', result.stdout)
        self.assertIn('12 full + 4 patch jobs', result.stdout)
        self.assertNotIn('bilingual', result.stdout)
        self.assertNotIn('PYTHON_LEGACY', result.stdout)
        runner = (Path(__file__).parents[1] / 'run_phase2.sh').read_text()
        self.assertEqual(1, runner.count('\n        run-group)'))

    def test_prepare_runtime_is_a_non_installing_setup_mode(self):
        root = Path(__file__).parents[1]
        runner = (root / 'run_phase2.sh').read_text()
        setup = (root / 'scripts' / 'setup_server_envs.sh').read_text()
        self.assertIn(
            'prepare-runtime) bash "$SCRIPT_DIR/scripts/setup_server_envs.sh" prepare-runtime',
            runner,
        )
        self.assertIn("MODE=\"${1:-install}\"", setup)
        self.assertIn("prepare-runtime)", setup)
        self.assertIn("require_env \"$VLLM_ENV\"", setup)
        self.assertIn("require_env \"$DEEPSEEK_ENV\"", setup)
        self.assertIn("require_env \"$VINTERN_ENV\"", setup)
        self.assertIn("require_env \"$HUATUO_ENV\"", setup)
        self.assertIn("Skipping package installation", setup)

    def test_server_profiles_cover_every_runtime_backend(self):
        root = Path(__file__).parents[1]
        profiles = root / 'requirements' / 'server'
        vllm = (profiles / 'vllm-blackwell.txt').read_text()
        self.assertIn('vllm==0.28.0', vllm)
        self.assertIn('torch==2.13.0', vllm)
        self.assertIn('transformers==5.17.0', vllm)
        int8 = (profiles / 'deepseek-int8-blackwell.txt').read_text()
        self.assertIn('bitsandbytes==0.49.0', int8)
        self.assertIn('transformers==4.38.2', int8)
        self.assertIn('torch==2.8.0', int8)
        self.assertNotIn('xformers', int8)
        vintern = (profiles / 'vintern-blackwell.txt').read_text()
        self.assertIn('transformers==4.42.3', vintern)
        self.assertIn('torch==2.8.0', vintern)
        huatuo = (profiles / 'huatuo-blackwell.txt').read_text()
        self.assertIn('transformers==4.37.2', huatuo)
        self.assertIn('torch==2.8.0', huatuo)
        setup = (root / 'scripts' / 'setup_server_envs.sh').read_text()
        self.assertNotIn('flash-attn', setup)
        self.assertNotIn('nvcc', setup)
        self.assertIn('write_export PYTHON_DEEPSEEK_VLLM "$VLLM_PYTHON"', setup)
        self.assertIn('bash "$KIT_DIR/run_phase2.sh" doctor', setup)
        runner = (root / 'run_phase2.sh').read_text()
        self.assertIn('exec bash "$SCRIPT_DIR/run_phase2.sh" all', runner)
        doctor = (root / 'scripts' / 'check_server_env.py').read_text()
        self.assertIn('DeepSeek Blackwell attention patch is missing', doctor)
        self.assertIn('Huatuo Blackwell attention patch is missing', doctor)

    def test_selected_qwen_models_use_bounded_deterministic_generation(self):
        config = (Path(__file__).parents[1] / 'vlmeval' / 'config.py').read_text()
        for model in ('Qwen3-VL-8B-Instruct', 'Qwen3.5-35B-A3B'):
            block = config.split(f'"{model}": partial(', 1)[1].split('\n    ),', 1)[0]
            self.assertIn('use_vllm=True', block)
            self.assertIn('temperature=0.0', block)
            self.assertIn('max_new_tokens=512', block)
            self.assertIn('presence_penalty=0.0', block)

        adapter = (Path(__file__).parents[1] / 'vlmeval' / 'vlm' / 'qwen3_vl' / 'model.py').read_text()
        self.assertIn("return {'enable_thinking': False}", adapter)


if __name__ == '__main__':
    unittest.main()
