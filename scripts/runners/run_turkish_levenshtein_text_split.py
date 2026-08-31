#!/usr/bin/env python3
"""
Run Turkish Levenshtein TEXT-SPLIT MPI benchmarks.
2 runs for each configuration, keeping the best result.
MPI-1, MPI-2, MPI-4, MPI-8 with all file sizes.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent.parent
NUM_RUNS = 2


def run_benchmark(num_procs: int, run_num: int):
    print(f"\n{'='*80}")
    print(f"RUN {run_num}/{NUM_RUNS} - MPI-{num_procs}")
    print(f"{'='*80}")
    sys.stdout.flush()

    script = BASE / "scripts/benchmarks_mpi/mpi_text_split_turkish_levenshtein.py"
    cmd = ["mpiexec", "-n", str(num_procs), "python", "-u", str(script)]
    subprocess.run(cmd, cwd=str(BASE))

    # Read results
    results_file = BASE / f"results/tr_text_split_levenshtein_mpi{num_procs}.json"
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    print("=" * 80)
    print("TURKISH LEVENSHTEIN TEXT-SPLIT BENCHMARK")
    print("2 runs per configuration, keeping best results")
    print("=" * 80)

    all_best_results = {}

    for num_procs in [1, 2, 4, 8]:
        best_results = None
        best_total_time = float('inf')

        for run_num in range(1, NUM_RUNS + 1):
            results = run_benchmark(num_procs, run_num)
            if results:
                # Calculate total time for comparison
                total_time = sum(r['total_time_s'] for r in results.get('results', []))
                print(f"\n  Run {run_num} total time: {total_time:.3f}s")

                if total_time < best_total_time:
                    best_total_time = total_time
                    best_results = results
                    print(f"  -> New best!")

        if best_results:
            all_best_results[str(num_procs)] = best_results

    # Save combined best results
    print("\n" + "=" * 80)
    print("FINAL BEST RESULTS")
    print("=" * 80)

    combined = {"description": "Turkish Levenshtein Text-Split - Best of 2 runs", "results": {}}

    for procs, data in all_best_results.items():
        combined["results"][procs] = data.get("results", [])
        print(f"\nMPI-{procs}:")
        print(f"{'Size':<8} | {'ms/word':<10} | {'Accuracy':<10}")
        print("-" * 35)
        for r in data.get("results", []):
            print(f"{r['size']:<8} | {r['ms_per_word']:<10.2f} | {r['accuracy_pct']:<10.1f}%")

    results_file = BASE / "results/tr_text_split_levenshtein_all.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nCombined results saved to: {results_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - ms/word")
    print("=" * 80)
    print(f"{'Size':<8} | {'MPI-1':<10} | {'MPI-2':<10} | {'MPI-4':<10} | {'MPI-8':<10}")
    print("-" * 60)

    sizes = ["250", "500", "1K", "1500", "2K", "2500", "3500", "5K"]
    for sz in sizes:
        row = [sz]
        for procs in ["1", "2", "4", "8"]:
            ms = 0
            for r in combined["results"].get(procs, []):
                if r['size'] == sz:
                    ms = r['ms_per_word']
                    break
            row.append(f"{ms:.2f}")
        print(f"{row[0]:<8} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10} | {row[4]:<10}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
