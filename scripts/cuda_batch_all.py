#!/usr/bin/env python3
"""
Run CUDA Batch benchmarks_mpi on 1K and 2K files.
"""

import os
import sys
import json
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spell_checker.cuda_batch import CUDABatchChecker, load_dictionary

import cupy as cp


def run_benchmark(checker, typos, ground_truth, algorithm, algo_name, lang, batch_size=64, progress_interval=200):
    """Run batch benchmark."""
    print(f"\n--- CUDA Batch {algo_name} + {lang} ({len(typos)} words, batch={batch_size}) ---")
    sys.stdout.flush()

    correct = 0
    total_time = 0

    for batch_start in range(0, len(typos), batch_size):
        batch_end = min(batch_start + batch_size, len(typos))
        batch_typos = typos[batch_start:batch_end]

        start = time.perf_counter()
        results = checker.find_corrections_batch(batch_typos, algorithm)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        total_time += elapsed

        for i, (best_word, best_dist) in enumerate(results):
            typo = batch_typos[i]
            if typo in ground_truth and best_word == ground_truth[typo]:
                correct += 1

        if (batch_end) % progress_interval == 0 or batch_end == len(typos):
            avg_ms = total_time / batch_end * 1000
            print(f"  Progress: {batch_end}/{len(typos)}, elapsed: {total_time:.2f}s, avg: {avg_ms:.3f}ms/word")
            sys.stdout.flush()

    ms_per_word = total_time / len(typos) * 1000
    accuracy = correct / len(typos) * 100

    print(f"  Completed: {len(typos)} words in {total_time:.2f}s, "
          f"{correct}/{len(typos)} correct ({accuracy:.1f}%), {ms_per_word:.3f}ms/word")
    sys.stdout.flush()

    return {
        'algorithm': algo_name,
        'language': lang,
        'num_words': len(typos),
        'batch_size': batch_size,
        'total_time_s': total_time,
        'ms_per_word': ms_per_word,
        'correct': correct,
        'accuracy_pct': accuracy
    }


def run_size_benchmark(base, size, checker_mk, checker_en):
    """Run benchmark for a specific size."""
    print(f"\n{'='*80}")
    print(f"CUDA BATCH BENCHMARK ({size} words)")
    print(f"{'='*80}")

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
        r = run_benchmark(checker_mk, typos_mk, gt_mk, algo_key, algo_name, 'MK', 64, progress_interval)
        results.append(r)
        r = run_benchmark(checker_en, typos_en, gt_en, algo_key, algo_name, 'EN', 64, progress_interval)
        results.append(r)

    return results


def main():
    print("=" * 100)
    print("CUDA BATCH BENCHMARK (1K and 2K words)")
    print("=" * 100)

    base = Path(__file__).parent.parent

    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"\nCUDA device: {device_name}")

    print("\nLoading dictionaries...")
    dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
    print(f"  MK: {len(dict_mk):,} words, EN: {len(dict_en):,} words")

    print("\nInitializing CUDA Batch checkers (dictionary on GPU)...")
    checker_mk = CUDABatchChecker(dict_mk)
    checker_en = CUDABatchChecker(dict_en)
    print("  Done")

    # Warmup
    print("\nWarming up kernels...")
    with open(base / "data/ground_truth/mk_hunspell_corrections_500.json", 'r', encoding='utf-8') as f:
        gt_warmup = json.load(f)
    typos_warmup = list(gt_warmup.keys())[:10]
    _ = checker_mk.find_corrections_batch(typos_warmup, 'levenshtein')
    _ = checker_mk.find_corrections_batch(typos_warmup, 'damerau')
    _ = checker_mk.find_corrections_batch(typos_warmup, 'myers')
    cp.cuda.Stream.null.synchronize()
    print("  Done")
    sys.stdout.flush()

    all_results = {}

    # Run 1K benchmark
    results_1k = run_size_benchmark(base, '1K', checker_mk, checker_en)
    all_results['1K'] = results_1k

    output_1k = {
        'description': 'CUDA Batch benchmark (1000 words)',
        'device': device_name,
        'batch_size': 64,
        'results': results_1k
    }
    with open(base / "results/cuda_batch_1K.json", 'w', encoding='utf-8') as f:
        json.dump(output_1k, f, indent=2, ensure_ascii=False)
    print(f"\n1K results saved to: results/cuda_batch_1K.json")

    # Run 2K benchmark
    results_2k = run_size_benchmark(base, '2K', checker_mk, checker_en)
    all_results['2K'] = results_2k

    output_2k = {
        'description': 'CUDA Batch benchmark (2000 words)',
        'device': device_name,
        'batch_size': 64,
        'results': results_2k
    }
    with open(base / "results/cuda_batch_2K.json", 'w', encoding='utf-8') as f:
        json.dump(output_2k, f, indent=2, ensure_ascii=False)
    print(f"\n2K results saved to: results/cuda_batch_2K.json")

    # Load 500 results
    with open(base / "results/cuda_batch_500.json", 'r', encoding='utf-8') as f:
        results_500 = json.load(f)['results']

    # Load per-word CUDA and C baseline for comparison
    per_word_lookup = {}
    c_baseline_lookup = {}

    for size in ['500', '1K', '2K']:
        try:
            with open(base / f"results/cuda_global_{size}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)['results']
                for r in data:
                    per_word_lookup[(size, r['algorithm'], r['language'])] = r['ms_per_word']
        except FileNotFoundError:
            pass

        try:
            with open(base / f"results/c_baseline_{size}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)['results']
                for r in data:
                    c_baseline_lookup[(size, r['algorithm'], r['language'])] = r['ms_per_word']
        except FileNotFoundError:
            pass

    # Build batch lookup
    batch_lookup = {'500': {}, '1K': {}, '2K': {}}
    for r in results_500:
        batch_lookup['500'][(r['algorithm'], r['language'])] = r['ms_per_word']
    for r in results_1k:
        batch_lookup['1K'][(r['algorithm'], r['language'])] = r['ms_per_word']
    for r in results_2k:
        batch_lookup['2K'][(r['algorithm'], r['language'])] = r['ms_per_word']

    # Print comparison table
    print("\n" + "=" * 140)
    print("CUDA BATCH - ALL SIZES COMPARISON")
    print("=" * 140)
    print(f"{'Algorithm':<20} | {'Lang':<4} | {'500w Batch':<12} | {'1Kw Batch':<12} | {'2Kw Batch':<12} | {'500w Per-word':<14} | {'500w C':<10}")
    print("-" * 140)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            b_500 = batch_lookup['500'].get((algo, lang), 0)
            b_1k = batch_lookup['1K'].get((algo, lang), 0)
            b_2k = batch_lookup['2K'].get((algo, lang), 0)
            pw_500 = per_word_lookup.get(('500', algo, lang), 0)
            c_500 = c_baseline_lookup.get(('500', algo, lang), 0)

            print(f"{algo:<20} | {lang:<4} | {b_500:>8.3f}ms   | {b_1k:>8.3f}ms   | {b_2k:>8.3f}ms   | {pw_500:>10.2f}ms   | {c_500:>6.2f}ms")

    print("-" * 140)

    # Speedup table
    print("\n" + "=" * 100)
    print("CUDA BATCH SPEEDUP vs C BASELINE")
    print("=" * 100)
    print(f"{'Algorithm':<20} | {'Lang':<4} | {'500w':<10} | {'1Kw':<10} | {'2Kw':<10}")
    print("-" * 100)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            speedups = []
            for size in ['500', '1K', '2K']:
                batch_ms = batch_lookup[size].get((algo, lang), 0)
                c_ms = c_baseline_lookup.get((size, algo, lang), 0)
                speedup = c_ms / batch_ms if batch_ms > 0 else 0
                speedups.append(speedup)

            print(f"{algo:<20} | {lang:<4} | {speedups[0]:>6.1f}x   | {speedups[1]:>6.1f}x   | {speedups[2]:>6.1f}x")

    print("-" * 100)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
