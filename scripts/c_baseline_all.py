#!/usr/bin/env python3
"""
Run C Baseline (Numba JIT) benchmarks_mpi on 1K and 2K files.
"""

import os
import sys
import json
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spell_checker.c_baseline import CBaselineChecker, load_dictionary


def run_benchmark(checker, typos, ground_truth, algorithm, algo_name, lang, progress_interval=100):
    """Run C baseline benchmark."""
    print(f"\n--- C Baseline (Numba JIT) {algo_name} + {lang} ({len(typos)} words) ---")
    sys.stdout.flush()

    correct = 0
    total_time = 0

    for i, typo in enumerate(typos):
        start = time.perf_counter()
        best_word, best_dist = checker.find_correction(typo, algorithm)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if typo in ground_truth and best_word == ground_truth[typo]:
            correct += 1

        if (i + 1) % progress_interval == 0:
            avg_ms = total_time / (i + 1) * 1000
            print(f"  Progress: {i+1}/{len(typos)}, elapsed: {total_time:.1f}s, avg: {avg_ms:.2f}ms/word")
            sys.stdout.flush()

    ms_per_word = total_time / len(typos) * 1000
    accuracy = correct / len(typos) * 100

    print(f"  Completed: {len(typos)} words in {total_time:.1f}s, "
          f"{correct}/{len(typos)} correct ({accuracy:.1f}%), {ms_per_word:.2f}ms/word")
    sys.stdout.flush()

    return {
        'algorithm': algo_name,
        'language': lang,
        'num_words': len(typos),
        'total_time_s': total_time,
        'ms_per_word': ms_per_word,
        'correct': correct,
        'accuracy_pct': accuracy
    }


def run_size_benchmark(base, size, checker_mk, checker_en):
    """Run benchmark for a specific size."""
    print(f"\n{'='*80}")
    print(f"C BASELINE (Numba JIT) BENCHMARK ({size} words)")
    print(f"{'='*80}")

    # Load ground truth
    with open(base / f"data/ground_truth/mk_hunspell_corrections_{size}.json", 'r', encoding='utf-8') as f:
        gt_mk = json.load(f)
    with open(base / f"data/ground_truth/en_hunspell_corrections_{size}.json", 'r', encoding='utf-8') as f:
        gt_en = json.load(f)

    typos_mk = list(gt_mk.keys())
    typos_en = list(gt_en.keys())
    print(f"  MK typos: {len(typos_mk)}, EN typos: {len(typos_en)}")
    sys.stdout.flush()

    algorithms = [
        ('levenshtein', 'Levenshtein'),
        ('damerau', 'Damerau-Levenshtein'),
        ('myers', 'Myers Bit-Vector'),
    ]

    results = []
    progress_interval = 200 if int(size.replace('K', '000')) >= 1000 else 100

    for algo_key, algo_name in algorithms:
        r = run_benchmark(checker_mk, typos_mk, gt_mk, algo_key, algo_name, 'MK', progress_interval)
        results.append(r)
        r = run_benchmark(checker_en, typos_en, gt_en, algo_key, algo_name, 'EN', progress_interval)
        results.append(r)

    return results


def main():
    print("=" * 100)
    print("C BASELINE (Numba JIT) BENCHMARK (1K and 2K words)")
    print("=" * 100)

    base = Path(__file__).parent.parent

    # Load dictionaries ONCE
    print("\nLoading dictionaries...")
    dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
    print(f"  MK: {len(dict_mk):,} words, EN: {len(dict_en):,} words")

    # Initialize checkers (pre-encode dictionaries ONCE)
    print("\nInitializing C baseline (pre-encoding dictionaries)...")
    checker_mk = CBaselineChecker(dict_mk)
    checker_en = CBaselineChecker(dict_en)
    print("  Done")

    # Warmup JIT
    print("\nWarming up JIT compilation...")
    _ = checker_mk.find_correction("test", "levenshtein")
    _ = checker_mk.find_correction("test", "damerau")
    _ = checker_mk.find_correction("test", "myers")
    print("  Done")
    sys.stdout.flush()

    all_results = {}

    # Run 1K benchmark
    results_1k = run_size_benchmark(base, '1K', checker_mk, checker_en)
    all_results['1K'] = results_1k

    # Save 1K results
    output_1k = {
        'description': 'C Baseline (Numba JIT) benchmark (1000 words)',
        'results': results_1k
    }
    with open(base / "results/c_baseline_1K.json", 'w', encoding='utf-8') as f:
        json.dump(output_1k, f, indent=2, ensure_ascii=False)
    print(f"\n1K results saved to: results/c_baseline_1K.json")

    # Run 2K benchmark
    results_2k = run_size_benchmark(base, '2K', checker_mk, checker_en)
    all_results['2K'] = results_2k

    # Save 2K results
    output_2k = {
        'description': 'C Baseline (Numba JIT) benchmark (2000 words)',
        'results': results_2k
    }
    with open(base / "results/c_baseline_2K.json", 'w', encoding='utf-8') as f:
        json.dump(output_2k, f, indent=2, ensure_ascii=False)
    print(f"\n2K results saved to: results/c_baseline_2K.json")

    # Load 500 results for comparison
    with open(base / "results/c_baseline_500.json", 'r', encoding='utf-8') as f:
        results_500 = json.load(f)['results']

    # Load CUDA results for comparison
    cuda_lookup = {}
    for size in ['500', '1K', '2K']:
        try:
            with open(base / f"results/cuda_global_{size}.json", 'r', encoding='utf-8') as f:
                cuda_data = json.load(f)['results']
                for r in cuda_data:
                    cuda_lookup[(size, r['algorithm'], r['language'])] = r['ms_per_word']
        except FileNotFoundError:
            pass

    # Build lookup
    ms_lookup = {'500': {}, '1K': {}, '2K': {}}
    for r in results_500:
        ms_lookup['500'][(r['algorithm'], r['language'])] = r['ms_per_word']
    for r in results_1k:
        ms_lookup['1K'][(r['algorithm'], r['language'])] = r['ms_per_word']
    for r in results_2k:
        ms_lookup['2K'][(r['algorithm'], r['language'])] = r['ms_per_word']

    # Print final comparison table
    print("\n" + "=" * 120)
    print("C BASELINE (Numba JIT) - SIZE COMPARISON")
    print("=" * 120)
    print(f"{'Algorithm':<20} | {'Lang':<4} | {'500w (ms)':<10} | {'1Kw (ms)':<10} | {'2Kw (ms)':<10} | {'Consistent?':<12}")
    print("-" * 120)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            ms_500 = ms_lookup['500'].get((algo, lang), 0)
            ms_1k = ms_lookup['1K'].get((algo, lang), 0)
            ms_2k = ms_lookup['2K'].get((algo, lang), 0)

            # Check consistency (within 50% of each other)
            if ms_500 > 0 and ms_1k > 0 and ms_2k > 0:
                avg = (ms_500 + ms_1k + ms_2k) / 3
                max_diff = max(abs(ms_500 - avg), abs(ms_1k - avg), abs(ms_2k - avg))
                consistent = "Yes" if max_diff / avg < 0.5 else "No"
            else:
                consistent = "N/A"

            print(f"{algo:<20} | {lang:<4} | {ms_500:>8.2f}ms | {ms_1k:>8.2f}ms | {ms_2k:>8.2f}ms | {consistent:<12}")

    print("-" * 120)

    # Print C vs CUDA comparison
    print("\n" + "=" * 120)
    print("C BASELINE vs CUDA COMPARISON")
    print("=" * 120)
    print(f"{'Algorithm':<20} | {'Lang':<4} | {'Size':<6} | {'C (ms)':<10} | {'CUDA (ms)':<10} | {'CUDA speedup':<12}")
    print("-" * 120)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            for size in ['500', '1K', '2K']:
                c_ms = ms_lookup[size].get((algo, lang), 0)
                cuda_ms = cuda_lookup.get((size, algo, lang), 0)
                speedup = c_ms / cuda_ms if cuda_ms > 0 else 0
                print(f"{algo:<20} | {lang:<4} | {size:<6} | {c_ms:>8.2f}ms | {cuda_ms:>8.2f}ms | {speedup:>10.1f}x")

    print("-" * 120)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
