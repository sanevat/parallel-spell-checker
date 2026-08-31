#!/usr/bin/env python3
"""Run Turkish Damerau-Levenshtein TEXT-SPLIT for all sizes and process counts.
Does 2 runs, compares results, and saves the better one."""

import subprocess
import sys
import json
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent
SIZES = ["250", "500", "1K", "1500", "2K", "2500", "3500", "5K"]
PROCS = [1, 2, 4, 8]
NUM_RUNS = 2

RESULTS_DIR = BASE / "results" / "text_split"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_single_benchmark(num_procs: int, size: str) -> dict:
    """Run a single benchmark and return the result from temp file."""
    cmd = ["mpiexec", "-n", str(num_procs), "python", "-u",
           str(BASE / "scripts/benchmarks_mpi/mpi_text_split_turkish_damerau.py"), size]
    subprocess.run(cmd, cwd=str(BASE), capture_output=True)

    temp_file = RESULTS_DIR / f"tr_damerau_text_split_{size}_temp.json"
    if temp_file.exists():
        with open(temp_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def compare_results(run1: dict, run2: dict) -> dict:
    """Compare two runs and return the better one (lower total_time_s)."""
    if run1 is None:
        return run2
    if run2 is None:
        return run1

    time1 = run1["results"][0]["total_time_s"]
    time2 = run2["results"][0]["total_time_s"]

    if time1 <= time2:
        return run1
    return run2


# Collect all best results
all_best_results = []

for num_procs in PROCS:
    for size in SIZES:
        print(f"\n{'='*60}")
        print(f"MPI-{num_procs} | Size: {size}")
        print(f"{'='*60}")

        best_result = None
        best_time = float('inf')

        for run_num in range(1, NUM_RUNS + 1):
            print(f"\n  [RUN {run_num}/{NUM_RUNS}]")
            result = run_single_benchmark(num_procs, size)

            if result:
                time_s = result["results"][0]["total_time_s"]
                ms_per_word = result["results"][0]["ms_per_word"]
                print(f"    Time: {time_s:.3f}s | {ms_per_word:.2f}ms/word")

                if time_s < best_time:
                    best_time = time_s
                    best_result = result

        if best_result:
            print(f"\n  BEST: {best_time:.3f}s")
            all_best_results.append(best_result["results"][0])

# Save final best results
final_file = RESULTS_DIR / "tr_damerau_best_results.json"
with open(final_file, 'w', encoding='utf-8') as f:
    json.dump({"runs_per_config": NUM_RUNS, "results": all_best_results}, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print(f"DONE! Best results saved to: {final_file}")
print(f"Total configurations: {len(all_best_results)}")
print("="*60)
