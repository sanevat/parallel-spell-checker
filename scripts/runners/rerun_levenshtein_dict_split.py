#!/usr/bin/env python3
"""
Run Dict-Split Levenshtein for 1, 2, 4, 8 procs on all datasets, 2 times each.
Compares results and saves minimum to JSON files in results/dict_split/.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent

DATASETS = [
    ('500', 'dict_split_500.json'),
    ('1K', 'dict_split_1K.json'),
    ('1500', 'dict_split_1500.json'),
    ('2K', 'dict_split_2K.json'),
    ('2500', 'dict_split_2500.json'),
    ('3500', 'dict_split_3500.json'),
    ('5000', 'dict_split_5000.json'),
]


def run_dict_split(num_procs, dataset_size):
    """Run Dict-Split benchmark and return results."""
    cmd = ["mpiexec", "-n", str(num_procs), "python", "-u",
           str(BASE / "scripts" / "dict_split_worker.py"), dataset_size]
    subprocess.run(cmd, cwd=str(BASE))

    temp_file = BASE / "results" / "dict_split_temp.json"
    if temp_file.exists():
        with open(temp_file, 'r', encoding='utf-8') as f:
            return json.load(f)['results']
    return []


def main():
    print("=" * 80)
    print("Dict-Split Levenshtein Benchmark (all datasets)")
    print("1, 2, 4, 8 procs | MK + EN | 2 runs each")
    print("=" * 80)

    # Create results/dict_split directory
    results_dir = BASE / "results" / "dict_split"
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset_size, json_file in DATASETS:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_size} words")
        print(f"{'='*80}")

        all_runs = {1: {'MK': [], 'EN': []}, 2: {'MK': [], 'EN': []}, 4: {'MK': [], 'EN': []}, 8: {'MK': [], 'EN': []}}

        for run in range(1, 3):
            print(f"\n--- Run {run}/2 ---")

            for num_procs in [1, 2, 4, 8]:
                print(f"  MPI-{num_procs}:", end=" ", flush=True)
                results = run_dict_split(num_procs, dataset_size)

                for r in results:
                    all_runs[num_procs][r['language']].append(r)
                    print(f"{r['language']}={r['total_time_s']:.1f}s", end=" ")
                print()

        # Compare and select best
        print(f"\n  Comparison:")
        best_results = {1: [], 2: [], 4: [], 8: []}

        for num_procs in [1, 2, 4, 8]:
            for lang in ['MK', 'EN']:
                runs = all_runs[num_procs][lang]
                if len(runs) == 2:
                    t1, t2 = runs[0]['total_time_s'], runs[1]['total_time_s']
                    diff_pct = abs(t1 - t2) / min(t1, t2) * 100
                    best = runs[0] if t1 <= t2 else runs[1]
                    best_results[num_procs].append(best)
                    status = "OK" if diff_pct < 15 else "DIFF!"
                    print(f"    {num_procs}p {lang}: {t1:.1f}s vs {t2:.1f}s ({diff_pct:.1f}% diff) [{status}] -> {best['total_time_s']:.1f}s")

        # Create and save JSON file
        all_data = {
            'description': f'DICT-SPLIT benchmark for Levenshtein ({dataset_size} words, 2 runs, best kept)',
            'dict_split': {}
        }

        for num_procs in [1, 2, 4, 8]:
            procs_key = str(num_procs)
            all_data['dict_split'][procs_key] = best_results[num_procs]

        results_file = results_dir / json_file
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {results_file}")

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY - Dict-Split Levenshtein (Best of 2 runs)")
    print("=" * 100)
    print(f"{'Dataset':<10} | {'Lang':<4} | {'1 Proc':>12} | {'2 Proc':>12} | {'4 Proc':>12} | {'8 Proc':>12} | {'Speedup':>8}")
    print("-" * 100)

    for dataset_size, json_file in DATASETS:
        results_file = results_dir / json_file
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for lang in ['MK', 'EN']:
                row = f"{dataset_size:<10} | {lang:<4} |"
                time_1 = 0
                for np in ['1', '2', '4', '8']:
                    for r in data['dict_split'].get(np, []):
                        if r['language'] == lang:
                            row += f" {r['total_time_s']:>10.2f}s |"
                            if np == '1':
                                time_1 = r['total_time_s']
                            if np == '8':
                                speedup = time_1 / r['total_time_s'] if r['total_time_s'] > 0 else 0
                                row += f" {speedup:>6.2f}x"
                print(row)

    print("-" * 100)
    print(f"\nResults saved to: {results_dir}")
    print("=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
