#!/usr/bin/env python3
"""
Run TEXT-SPLIT MPI benchmarks_mpi for 3500 test files.
MPI-2, MPI-4, MPI-8 only (no sequential).
"""

import os
import sys
import json
import subprocess
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent


def main():
    print("=" * 100)
    print("TEXT-SPLIT MPI BENCHMARK (3500 words)")
    print("=" * 100)
    print("\nRunning TEXT-SPLIT MPI-2, MPI-4, MPI-8")
    sys.stdout.flush()

    all_results = {
        'description': 'TEXT-SPLIT benchmark for 3500 words',
        'text_split': {}
    }

    for num_procs in [2, 4, 8]:
        print(f"\n{'='*80}")
        print(f"RUNNING TEXT-SPLIT MPI-{num_procs}")
        print(f"{'='*80}")
        sys.stdout.flush()

        cmd = ["mpiexec", "-n", str(num_procs), "python", "-u",
               str(BASE / "spell_checker" / "mpi_text_split_3500.py")]
        subprocess.run(cmd, cwd=str(BASE))

        temp_file = BASE / "results" / "text_split_3500_temp.json"
        if temp_file.exists():
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_results['text_split'][str(num_procs)] = data.get('results', [])

    # Build lookup
    text_results = {}
    for procs in ['2', '4', '8']:
        for r in all_results['text_split'].get(procs, []):
            key = (r['algorithm'], r['language'], int(procs))
            text_results[key] = r['ms_per_word']

    # Print final table
    print("\n" + "=" * 100)
    print("FINAL TEXT-SPLIT RESULTS (3500 words)")
    print("=" * 100)
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Text-2':<10} | {'Text-4':<10} | {'Text-8':<10}")
    print("-" * 100)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            text2 = text_results.get((algo, lang, 2), 0)
            text4 = text_results.get((algo, lang, 4), 0)
            text8 = text_results.get((algo, lang, 8), 0)
            print(f"{algo:<22} | {lang:<4} | {text2:>8.1f}ms | {text4:>8.1f}ms | {text8:>8.1f}ms")

    print("-" * 100)

    # Save results
    results_file = BASE / "results" / "text_split_3500.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
