#!/usr/bin/env python3
"""
Run Damerau-Levenshtein DICT-SPLIT benchmark for Turkish with 2 runs, taking the best result.
Tests: 250, 500, 1K, 1.5K, 2K, 2.5K, 3.5K, 5K words with 1, 2, 4, 8 processes.
"""

import subprocess
import json
import sys
from pathlib import Path

NUM_RUNS = 2
SIZES = ["250", "500", "1K", "1500", "2K", "2500", "3500", "5K"]
PROCS = [1, 2, 4, 8]

base = Path(__file__).parent.parent
benchmark_script = base / "scripts" / "benchmarks_mpi" / "mpi_dict_split_turkish_damerau.py"
results_dir = base / "results" / "dict_split"
results_dir.mkdir(parents=True, exist_ok=True)

all_results = []

print("=" * 80)
print("DAMERAU-LEVENSHTEIN TURKISH BENCHMARK - DICT-SPLIT STRATEGY")
print(f"Runs per config: {NUM_RUNS} (best time kept)")
print(f"Sizes: {SIZES}")
print(f"Processes: {PROCS}")
print("=" * 80)

for size in SIZES:
    for num_procs in PROCS:
        best_result = None
        best_time = float('inf')

        print(f"\n>>> Testing {size} words with {num_procs} process(es)...")

        for run in range(1, NUM_RUNS + 1):
            print(f"    Run {run}/{NUM_RUNS}...", end=" ", flush=True)

            cmd = ["mpiexec", "-n", str(num_procs), sys.executable, str(benchmark_script), size]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                # Read temp file
                temp_file = results_dir / f"tr_damerau_dict_split_{size}_temp.json"
                if temp_file.exists():
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if data.get("results"):
                        run_result = data["results"][0]
                        run_time = run_result["total_time_s"]
                        print(f"{run_time:.2f}s", flush=True)

                        if run_time < best_time:
                            best_time = run_time
                            best_result = run_result
                else:
                    print("ERROR: temp file not found", flush=True)

            except subprocess.TimeoutExpired:
                print("TIMEOUT", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)

        if best_result:
            print(f"    Best: {best_time:.2f}s")
            all_results.append(best_result)

# Save final results
output_file = results_dir / "tr_damerau_dict_split_best_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({"runs_per_config": NUM_RUNS, "results": all_results}, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"DONE! Results saved to: {output_file}")
print(f"Total configurations: {len(all_results)}")
print("=" * 80)
