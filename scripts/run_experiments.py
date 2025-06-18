import subprocess
import os
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

class ExperimentRunner:
    def __init__(self, max_parallel=2):
        self.max_parallel = max_parallel
        self.results = {}
        self.lock = threading.Lock()

    def check_config_exists(self, config_name):
        """Verify that the configuration file exists"""
        config_path = Path(f"conf/experiments/{config_name}.yaml")
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False
        return True

    def run_single_experiment(self, config_name, experiment_name):
        print(f"\n🚀 Starting Experiment: {experiment_name}")
        print(f"Config: experiments/{config_name}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)

        if not self.check_config_exists(config_name):
            raise FileNotFoundError(f"Configuration experiments/{config_name}.yaml not found")

        start_time = time.time()
        exp_dir = Path(f"./experiments/{experiment_name}")
        exp_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                'python', 'scripts/model_training.py',
                f'--config-name=experiments/{config_name}',
                f'hydra.run.dir={exp_dir}/hydra_outputs'
            ]
            print(f"🔧 Running: {' '.join(cmd)}")
            env = os.environ.copy()
            env['HYDRA_FULL_ERROR'] = '1'  # Enable full stack traces
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env, check=True)
            training_time = time.time() - start_time

            print(f"✅ Training completed for {experiment_name} ({training_time:.1f}s)")

            eval_start = time.time()
            eval_cmd = ['python', 'scripts/model_evaluation.py']
            env['MODEL_PATH'] = str(exp_dir / 'final_model')
            env['VAL_PATH'] = 'outputs/val_dataset'
            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, env=env)
            eval_time = time.time() - eval_start

            metrics = self.parse_evaluation_output(eval_result.stdout)
            if not metrics:
                print(f"⚠️ No metrics parsed from evaluation output for {experiment_name}")

            experiment_result = {
                'status': 'success' if eval_result.returncode == 0 else 'failed',
                'training_time': training_time,
                'evaluation_time': eval_time,
                'total_time': training_time + eval_time,
                'metrics': metrics if metrics else {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'config_name': config_name,
                'output_dir': str(exp_dir),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'priority_fixes_applied': True,
                'eval_stderr': eval_result.stderr[-500:] if eval_result.returncode != 0 else ''
            }

            print(f"✅ {experiment_name} COMPLETED! Time: {training_time:.1f}s")
            if metrics:
                print(f"📊 ROUGE-L: {metrics.get('rougeL', 'N/A'):.4f}")

        except subprocess.CalledProcessError as e:
            experiment_result = {
                'status': 'failed',
                'error': e.stderr[-1000:],
                'training_time': time.time() - start_time,
                'config_name': config_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'priority_fixes_applied': True
            }
            print(f"❌ {experiment_name} FAILED! ({experiment_result['training_time']:.1f}s)")
            print(f"Error: {e.stderr[-500:]}")

        except Exception as e:
            experiment_result = {
                'status': 'error',
                'error': str(e),
                'config_name': config_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'priority_fixes_applied': True
            }
            print(f"💥 Exception in {experiment_name}: {e}")

        with self.lock:
            self.results[experiment_name] = experiment_result
        return experiment_result

    def parse_evaluation_output(self, output):
        """Parse evaluation metrics with robust error handling"""
        metrics = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        if not output or not isinstance(output, str):
            print("⚠️ Evaluation output is empty or invalid")
            return metrics

        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            try:
                if 'ROUGE-1:' in line:
                    score = float(re.search(r'ROUGE-1:\s*([\d.]+)', line).group(1))
                    metrics['rouge1'] = score
                elif 'ROUGE-2:' in line:
                    score = float(re.search(r'ROUGE-2:\s*([\d.]+)', line).group(1))
                    metrics['rouge2'] = score
                elif 'ROUGE-L:' in line:
                    score = float(re.search(r'ROUGE-L:\s*([\d.]+)', line).group(1))
                    metrics['rougeL'] = score
            except (AttributeError, ValueError) as e:
                print(f"⚠️ Failed to parse metric line: {line} ({e})")
        return metrics

    def run_experiment_batch(self, experiments, batch_name):
        print(f"🎯 Running Batch {batch_name}: {len(experiments)} experiments")
        print(f"🚀 Experiments in this batch:")
        for config, name in experiments:
            print(f"   - {name} ({config})")
        print("=" * 80)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_exp = {
                executor.submit(self.run_single_experiment, config, name): name
                for config, name in experiments
            }
            completed = 0
            for future in as_completed(future_to_exp):
                exp_name = future_to_exp[future]
                completed += 1
                try:
                    result = future.result()
                    status_emoji = "✅" if result['status'] == 'success' else "❌"
                    print(f"\n{status_emoji} [{completed}/{len(experiments)}] {exp_name}: {result['status']}")
                    if result.get('metrics'):
                        print(f"   📊 ROUGE-L: {result['metrics'].get('rougeL', 0):.4f}")
                    else:
                        print(f"   ⚠️ No metrics available for {exp_name}")
                except Exception as e:
                    print(f"💥 {exp_name} crashed: {e}")

        batch_time = time.time() - start_time
        print(f"\n🏁 Batch {batch_name} completed in {batch_time:.1f}s")
        return self.results

EXPERIMENT_BATCHES = {
    1: [
        ("baseline", "baseline"),
        ("fast", "fast_training")
    ],
    2: [
        ("aggressive", "aggressive_training"),
        ("data_augmented", "data_augmented")
    ],
    3: [
        ("balanced", "balanced_training"),
        ("quality", "quality_training")
    ]
}

def run_batch_experiments(batch_number=1):
    if batch_number not in EXPERIMENT_BATCHES:
        print(f"❌ Invalid batch number: {batch_number}")
        print(f"Available batches: {list(EXPERIMENT_BATCHES.keys())}")
        return

    experiments = EXPERIMENT_BATCHES[batch_number]
    print("🎯 ENHANCED PARALLEL EXPERIMENT RUNNER")
    print("=" * 70)
    print("🔧 PRIORITY FIXES INCLUDED:")
    print("✅ Simplified preprocessing")
    print("✅ Single-task training")
    print("✅ Consistent evaluation")
    print("✅ Optimized inference")
    print("✅ Enhanced config validation")
    print("=" * 70)

    runner = ExperimentRunner(max_parallel=2)
    results = runner.run_experiment_batch(experiments, batch_number)
    if results:
        try:
            best_exp_name = max(results, key=lambda x: results[x].get('metrics', {}).get('rougeL', 0))
            best_result = results[best_exp_name]
            print(f"\n🏆 BATCH {batch_number} WINNER: {best_exp_name}")
            if best_result.get('metrics'):
                print(f"📈 ROUGE-L Score: {best_result['metrics'].get('rougeL', 0):.4f}")
                print(f"📁 Model Location: {best_result['output_dir']}/final_model")
                print(f"⚙️ Config Used: conf/experiments/{best_result['config_name']}.yaml")
                print(f"⏱️ Training Time: {best_result['total_time']/60:.1f} minutes")
            else:
                print(f"⚠️ No metrics available for winner {best_exp_name}")
        except Exception as e:
            print(f"❌ Failed to determine batch winner: {e}")
    return results

if __name__ == "__main__":
    import sys
    batch_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    results = run_batch_experiments(batch_number)