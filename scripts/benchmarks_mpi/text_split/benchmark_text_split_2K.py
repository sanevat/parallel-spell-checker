#!/usr/bin/env python3
"""
Run TEXT-SPLIT MPI benchmarks_mpi for 2K test files.
Only MPI-2, MPI-4, MPI-8 (no sequential, no dict-split).
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent

# Sequential baseline (from sequential_2K.json)
SEQ_BASELINE = {
    ("Levenshtein", "MK"): 438.1,
    ("Levenshtein", "EN"): 322.2,
    ("Damerau-Levenshtein", "MK"): 478.9,
    ("Damerau-Levenshtein", "EN"): 460.0,
    ("Myers Bit-Vector", "MK"): 135.4,
    ("Myers Bit-Vector", "EN"): 120.2,
}

# Dict-split MPI-8 results (from mpi_2K.json)
DICT_SPLIT_8 = {
    ("Levenshtein", "MK"): 79.5,
    ("Levenshtein", "EN"): 60.9,
    ("Damerau-Levenshtein", "MK"): 88.6,
    ("Damerau-Levenshtein", "EN"): 86.9,
    ("Myers Bit-Vector", "MK"): 27.7,
    ("Myers Bit-Vector", "EN"): 24.1,
}


def main():
    print("=" * 100)
    print("TEXT-SPLIT MPI BENCHMARK (2000 words)")
    print("=" * 100)
    print("\nRunning TEXT-SPLIT MPI-2, MPI-4, MPI-8 only (no sequential, no dict-split)")
    sys.stdout.flush()

    all_results = {
        'description': 'TEXT-SPLIT benchmark for 2K words',
        'text_split': {}
    }

    # Run TEXT-SPLIT for 2, 4, 8 processes
    for num_procs in [2, 4, 8]:
        print(f"\n{'='*80}")
        print(f"RUNNING TEXT-SPLIT MPI-{num_procs}")
        print(f"{'='*80}")
        sys.stdout.flush()

        cmd = ["mpiexec", "-n", str(num_procs), "python", "-u",
               str(BASE / "spell_checker" / "mpi_text_split_2K.py")]
        subprocess.run(cmd, cwd=str(BASE))

        # Read results
        temp_file = BASE / "results" / "text_split_2K_temp.json"
        if temp_file.exists():
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_results['text_split'][str(num_procs)] = data.get('results', [])

    # Build lookup for text-split results
    text_results = {}
    for procs in ['2', '4', '8']:
        for r in all_results['text_split'].get(procs, []):
            key = (r['algorithm'], r['language'], int(procs))
            text_results[key] = r['ms_per_word']

    # Print final comparison table
    print("\n" + "=" * 120)
    print("FINAL COMPARISON: TEXT-SPLIT vs DICT-SPLIT (2000 words)")
    print("=" * 120)
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Seq':<8} | {'Text-2':<8} | {'Text-4':<8} | {'Text-8':<8} | {'Dict-8':<8} | {'Text Spd-8':<10}")
    print("-" * 120)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            seq_ms = SEQ_BASELINE.get((algo, lang), 0)
            text2 = text_results.get((algo, lang, 2), 0)
            text4 = text_results.get((algo, lang, 4), 0)
            text8 = text_results.get((algo, lang, 8), 0)
            dict8 = DICT_SPLIT_8.get((algo, lang), 0)

            text_speedup = seq_ms / text8 if text8 > 0 else 0

            print(f"{algo:<22} | {lang:<4} | {seq_ms:>6.1f}ms | {text2:>6.1f}ms | {text4:>6.1f}ms | {text8:>6.1f}ms | {dict8:>6.1f}ms | {text_speedup:>8.2f}x")

    print("-" * 120)

    # Print TEXT vs DICT comparison for MPI-8
    print("\n" + "=" * 100)
    print("TEXT-SPLIT vs DICT-SPLIT COMPARISON (MPI-8, 2000 words)")
    print("=" * 100)
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Dict-8':<10} | {'Text-8':<10} | {'Winner':<12} | {'Advantage':<10}")
    print("-" * 100)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            text8 = text_results.get((algo, lang, 8), 0)
            dict8 = DICT_SPLIT_8.get((algo, lang), 0)

            if text8 > 0 and dict8 > 0:
                if text8 < dict8:
                    winner = "TEXT-SPLIT"
                    advantage = (dict8 - text8) / dict8 * 100
                else:
                    winner = "DICT-SPLIT"
                    advantage = (text8 - dict8) / text8 * 100
            else:
                winner = "N/A"
                advantage = 0

            print(f"{algo:<22} | {lang:<4} | {dict8:>8.1f}ms | {text8:>8.1f}ms | {winner:<12} | {advantage:>7.1f}%")

    print("-" * 100)

    # Save results
    results_file = BASE / "results" / "text_split_2K.json"

    # Add sequential and dict-split baselines to saved results
    all_results['sequential_baseline'] = {f"{a}_{l}": v for (a, l), v in SEQ_BASELINE.items()}
    all_results['dict_split_8'] = {f"{a}_{l}": v for (a, l), v in DICT_SPLIT_8.items()}

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
