#!/usr/bin/env python3
"""
Re-run Damerau-Levenshtein for 1, 2, 4, 8 procs on all datase
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
    ('250', 'text_split_250.json'),
    ('500', 'text_split_500.json'),
    ('1K', 'text_split_1K.json'),
    ('1500', 'text_split_1500.json'),
    ('2K', 'text_split_2K.json'),
    ('2500', 'text_split_2500.json'),
    ('3500', 'text_split_3500.json'),
    ('5000', 'text_split_5000.json'),
]


def run_damerau_levenshtein(num_procs, dataset_size):
    """Run Damerau-Levenshtein benchmark and return results."""
    cmd = ["mpiexec", "-n", str(num_procs), "python", "-u",
           str(BASE / "scripts" / "damerau_levenshtein_worker.py"), dataset_size]
    subprocess.run(cmd, cwd=str(BASE))

    temp_file = BASE / "results" / "damerau_levenshtein_temp.json"
    if temp_file.exists():
        with open(temp_file, 'r', encoding='utf-8') as f:
            return json.load(f)['results']
    return []


def main():
    print("=" * 80)
    print("Damerau-Levenshtein Re-run (all datasets)")
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
                results = run_damerau_levenshtein(num_procs, dataset_size)

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

        # Update JSON file
        results_file = BASE / "results" / json_file
        with open(results_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        # Ensure text_split exists
        if 'text_split' not in all_data:
            all_data['text_split'] = {}

        for num_procs in [1, 2, 4, 8]:
            procs_key = str(num_procs)

            # Ensure the process key exists
            if procs_key not in all_data['text_split']:
                all_data['text_split'][procs_key] = []

            for best in best_results[num_procs]:
                # Find and update existing entry, or add new one
                found = False
                for i, entry in enumerate(all_data['text_split'][procs_key]):
                    if entry['algorithm'] == 'Damerau-Levenshtein' and entry['language'] == best['language']:
                        all_data['text_split'][procs_key][i] = best
                        found = True
                        break

                if not found:
                    all_data['text_split'][procs_key].append(best)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        print(f"  Updated: {json_file}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
