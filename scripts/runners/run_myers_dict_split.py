#!/usr/bin/env python3
"""
Run Myers Bit-Vector Dict-Split for 1, 2, 4, 8 procs on all datasets, 2 times each.
Compares results and saves minimum to dict_split JSON files.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent.parent

DATASETS = [
    ('250', 'dict_split_250.json'),
    ('500', 'dict_split_500.json'),
    ('1K', 'dict_split_1K.json'),
    ('1500', 'dict_split_1500.json'),
    ('2K', 'dict_split_2K.json'),
    ('2500', 'dict_split_2500.json'),
    ('3500', 'dict_split_3500.json'),
    ('5000', 'dict_split_5000.json'),
]


def run_myers(num_procs, dataset_size):
    """Run Myers Bit-Vector benchmark and return results."""
    worker_script = BASE / "scripts" / "workers" / "myers_dict_split_worker.py"
    cmd = ["mpiexec", "-n", str(num_procs), "python", "-u", str(worker_script), dataset_size]

    try:
        subprocess.run(cmd, cwd=str(BASE), timeout=3600)
    except subprocess.TimeoutExpired:
        print("TIMEOUT!")
        return []

    temp_file = BASE / "results" / "myers_dict_split_temp.json"
    if temp_file.exists():
        with open(temp_file, 'r', encoding='utf-8') as f:
            return json.load(f)['results']
    return []


def main():
    print("=" * 80)
    print("Myers Bit-Vector Dict-Split Benchmark")
    print("1, 2, 4, 8 procs | MK + EN | 2 runs each")
    print("=" * 80)

    for dataset_size, json_file in DATASETS:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_size} words")
        print(f"{'='*80}")

        all_runs = {1: {'MK': [], 'EN': []}, 2: {'MK': [], 'EN': []}, 4: {'MK': [], 'EN': []}, 8: {'MK': [], 'EN': []}}

        for run in range(1, 3):
            print(f"\n--- Run {run}/2 ---")

            for num_procs in [1, 2, 4, 8]:
                print(f"  MPI-{num_procs}:", end=" ", flush=True)
                results = run_myers(num_procs, dataset_size)

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
                    diff_pct = abs(t1 - t2) / min(t1, t2) * 100 if min(t1, t2) > 0 else 0
                    best = runs[0] if t1 <= t2 else runs[1]
                    best_results[num_procs].append(best)
                    status = "OK" if diff_pct < 15 else "DIFF!"
                    print(f"    {num_procs}p {lang}: {t1:.1f}s vs {t2:.1f}s ({diff_pct:.1f}% diff) [{status}] -> {best['total_time_s']:.1f}s")
                elif len(runs) == 1:
                    best_results[num_procs].append(runs[0])
                    print(f"    {num_procs}p {lang}: only 1 run -> {runs[0]['total_time_s']:.1f}s")

        # Update JSON file
        results_file = BASE / "results" / "dict_split" / json_file

        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        else:
            all_data = {"description": f"DICT-SPLIT benchmark for {dataset_size} words", "dict_split": {}}

        # Ensure dict_split exists
        if 'dict_split' not in all_data:
            all_data['dict_split'] = {}

        for num_procs in [1, 2, 4, 8]:
            procs_key = str(num_procs)

            # Ensure the process key exists
            if procs_key not in all_data['dict_split']:
                all_data['dict_split'][procs_key] = []

            for best in best_results[num_procs]:
                # Find and update existing entry, or add new one
                found = False
                for i, existing in enumerate(all_data['dict_split'][procs_key]):
                    if existing['algorithm'] == 'Myers Bit-Vector' and existing['language'] == best['language']:
                        all_data['dict_split'][procs_key][i] = best
                        found = True
                        break

                if not found:
                    all_data['dict_split'][procs_key].append(best)

        # Save updated JSON
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        print(f"\n  Updated: {results_file}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
