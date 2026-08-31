#!/usr/bin/env python3
"""
Runner script for Turkish Levenshtein Dict-Split benchmark.
2 runs, 1/2/4/8 processes, 250-5K words.
"""

import subprocess
import json
import sys
import time
from pathlib import Path

base_dir = Path(__file__).parent.parent
results_dir = base_dir / "results" / "dict_split"
results_dir.mkdir(parents=True, exist_ok=True)

# Configuration
RUNS_PER_CONFIG = 2
PROCESSES = [1, 2, 4, 8]
WORD_SIZES = ["250", "500", "1K", "1500", "2K", "2500", "3500", "5K"]

script_path = base_dir / "scripts" / "benchmarks_mpi" / "mpi_dict_split_turkish_levenshtein.py"

all_results = []

print("=" * 80)
print("TURKISH LEVENSHTEIN DICT-SPLIT BENCHMARK")
print(f"Runs per config: {RUNS_PER_CONFIG}")
print(f"Processes: {PROCESSES}")
print(f"Word sizes: {WORD_SIZES}")
print("=" * 80)

for word_size in WORD_SIZES:
    for num_procs in PROCESSES:
        print(f"\n>>> Running: {word_size} words, {num_procs} processes...")

        best_result = None
        best_time = float('inf')

        for run in range(RUNS_PER_CONFIG):
            print(f"  Run {run + 1}/{RUNS_PER_CONFIG}...")

            cmd = ["mpiexec", "-n", str(num_procs), sys.executable, str(script_path), word_size]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                if result.returncode != 0:
                    print(f"  ERROR: {result.stderr[:200]}")
                    continue

                # Read temp result file
                temp_file = results_dir / f"tr_levenshtein_dict_split_{word_size}_temp.json"
                if temp_file.exists():
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        temp_data = json.load(f)

                    run_result = temp_data['results'][0]
                    run_time = run_result['total_time_s']

                    print(f"    Time: {run_time:.2f}s, Accuracy: {run_result['accuracy_pct']:.1f}%")

                    if run_time < best_time:
                        best_time = run_time
                        best_result = run_result

            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT!")
                continue
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        if best_result:
            all_results.append(best_result)
            print(f"  Best time: {best_time:.2f}s")

# Save final results
output_file = results_dir / "tr_levenshtein_dict_split_best_results.json"
final_data = {
    "runs_per_config": RUNS_PER_CONFIG,
    "results": all_results
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"COMPLETED! Results saved to: {output_file}")
print(f"Total configurations: {len(all_results)}")
print("=" * 80)
