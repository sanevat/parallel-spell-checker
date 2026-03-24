#!/usr/bin/env python3
"""
Run full benchmark for 250-word test files (Sequential + MPI).
Uses existing test files and ground truth generated from dictionary.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector

BASE = Path(__file__).parent.parent


def load_dictionary(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def group_by_length(words: List[str]) -> Dict[int, List[str]]:
    by_len = defaultdict(list)
    for w in words:
        by_len[len(w)].append(w)
    return dict(by_len)


def get_candidates(word_len: int, by_len: Dict[int, List[str]], tol: int = 2) -> List[str]:
    candidates = []
    for length in range(max(1, word_len - tol), word_len + tol + 1):
        if length in by_len:
            candidates.extend(by_len[length])
    return candidates


def run_sequential_benchmark(typos_mk, typos_en, mk_by_len, en_by_len, gt_mk, gt_en):
    """Run sequential benchmark."""
    print("\n" + "=" * 80)
    print("SEQUENTIAL BENCHMARK")
    print("=" * 80)

    algorithms = [
        ("Levenshtein", levenshtein),
        ("Damerau-Levenshtein", damerau_levenshtein),
        ("Myers Bit-Vector", myers_bitvector),
    ]

    results = []

    for algo_name, algo_fn in algorithms:
        for lang, typos, by_len, gt in [
            ("MK", typos_mk, mk_by_len, gt_mk),
            ("EN", typos_en, en_by_len, gt_en)
        ]:
            print(f"\n--- Sequential {algo_name} + {lang} ({len(typos)} words) ---")
            sys.stdout.flush()

            correct = 0
            total_time = 0
            total_candidates = 0

            for i, typo in enumerate(typos):
                candidates = get_candidates(len(typo), by_len)
                total_candidates += len(candidates)

                start = time.perf_counter()
                best_word = ""
                best_dist = 999999
                for c in candidates:
                    d = algo_fn(typo, c)
                    if d < best_dist:
                        best_dist = d
                        best_word = c
                        if d == 0:
                            break
                elapsed = time.perf_counter() - start
                total_time += elapsed

                if typo in gt and best_word == gt[typo]:
                    correct += 1

                if (i + 1) % 25 == 0:
                    avg_ms = total_time / (i + 1) * 1000
                    print(f"  Progress: {i+1}/{len(typos)}, elapsed: {total_time:.1f}s, avg: {avg_ms:.1f}ms/word")
                    sys.stdout.flush()

            ms_per_word = total_time / len(typos) * 1000
            accuracy = correct / len(typos) * 100

            print(f"  Completed: {len(typos)} words in {total_time:.1f}s, "
                  f"{correct}/{len(typos)} correct ({accuracy:.1f}%)")
            sys.stdout.flush()

            results.append({
                'algorithm': algo_name,
                'language': lang,
                'num_procs': 1,
                'misspelled': len(typos),
                'total_time_s': total_time,
                'ms_per_word': ms_per_word,
                'correct': correct,
                'accuracy_pct': accuracy,
                'avg_candidates': total_candidates / len(typos)
            })

    return results


def main():
    print("=" * 80)
    print("BENCHMARK 250 WORDS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    dict_mk = load_dictionary(str(BASE / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(BASE / "data/dictionary/en_equal.txt"))

    mk_by_len = group_by_length(dict_mk)
    en_by_len = group_by_length(dict_en)

    # Load ground truth
    with open(BASE / "data/ground_truth/mk_hunspell_corrections_250.json", 'r', encoding='utf-8') as f:
        gt_mk = json.load(f)
    with open(BASE / "data/ground_truth/en_hunspell_corrections_250.json", 'r', encoding='utf-8') as f:
        gt_en = json.load(f)

    # Extract typos
    typos_mk = list(gt_mk.keys())
    typos_en = list(gt_en.keys())

    print(f"MK typos: {len(typos_mk)}, EN typos: {len(typos_en)}")
    print(f"MK dict: {len(dict_mk):,} words, EN dict: {len(dict_en):,} words")

    all_results = {
        'description': 'Scalability benchmark with 250 typos, 50K dict',
        'sequential': [],
        'mpi': {}
    }

    # Run sequential benchmark
    seq_results = run_sequential_benchmark(typos_mk, typos_en, mk_by_len, en_by_len, gt_mk, gt_en)
    all_results['sequential'] = seq_results

    # Build baseline for speedup calculation
    seq_baseline = {}
    for r in seq_results:
        seq_baseline[(r['algorithm'], r['language'])] = r['ms_per_word']

    # Save sequential results immediately
    results_file = BASE / "results" / "scalability_250.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSequential results saved to: {results_file}")

    # Run MPI benchmarks_mpi
    for num_procs in [2, 4, 8]:
        print(f"\n{'='*80}")
        print(f"RUNNING MPI-{num_procs} BENCHMARK")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Run MPI benchmark via subprocess
        mpi_script = BASE / "spell_checker" / "mpi_benchmark_250.py"
        cmd = ["mpiexec", "-n", str(num_procs), "python", "-u", str(mpi_script)]

        subprocess.run(cmd, cwd=str(BASE))

        # Read results
        temp_file = BASE / "results" / "mpi_250_temp.json"
        if temp_file.exists():
            with open(temp_file, 'r', encoding='utf-8') as f:
                mpi_data = json.load(f)
            all_results['mpi'][str(num_procs)] = mpi_data.get('results', [])

    # Save final results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print final comparison table
    print("\n" + "=" * 100)
    print("FINAL COMPARISON TABLE (250 words)")
    print("=" * 100)
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Seq(ms)':<8} | {'MPI-2':<8} | {'MPI-4':<8} | {'MPI-8':<8} | {'Speedup-8':<10}")
    print("-" * 100)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            seq_ms = seq_baseline.get((algo, lang), 0)

            mpi2_ms = 0
            mpi4_ms = 0
            mpi8_ms = 0

            for r in all_results['mpi'].get('2', []):
                if r['algorithm'] == algo and r['language'] == lang:
                    mpi2_ms = r['ms_per_word']
            for r in all_results['mpi'].get('4', []):
                if r['algorithm'] == algo and r['language'] == lang:
                    mpi4_ms = r['ms_per_word']
            for r in all_results['mpi'].get('8', []):
                if r['algorithm'] == algo and r['language'] == lang:
                    mpi8_ms = r['ms_per_word']

            speedup = seq_ms / mpi8_ms if mpi8_ms > 0 else 0

            print(f"{algo:<22} | {lang:<4} | {seq_ms:>8.1f} | {mpi2_ms:>8.1f} | {mpi4_ms:>8.1f} | {mpi8_ms:>8.1f} | {speedup:>8.2f}x")

    print("-" * 100)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
