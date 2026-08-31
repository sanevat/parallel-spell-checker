#!/usr/bin/env python3
"""
Comprehensive CUDA Batch vs C-Sequential Benchmark

Tests all combinations:
- Word counts: 250, 500, 1K, 1500, 2K, 2500, 3500, 5K
- Languages: MK, EN, TR
- Algorithms: Levenshtein, Damerau-Levenshtein, Myers Bit-Vector

CUDA Batch Implementation:
- Groups typos by length (1 kernel launch per group)
- 2D grid: blockIdx.y = typo index, threadIdx.x = candidate index
- Each thread computes one edit distance
- Argmin kernel finds best match per typo
- Dictionary stays on GPU (pre-loaded)

Runs multiple iterations and reports MEDIAN results for stability.

Outputs results to JSON and CSV for analysis.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import statistics

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spell_checker.cuda_batch import CUDABatchChecker
from spell_checker.c_baseline import CBaselineChecker

import cupy as cp

# Batch size for CUDA batch processing
BATCH_SIZE = 64

# Number of iterations for median calculation
NUM_ITERATIONS = 4


def load_dictionary(path: str) -> List[str]:
    """Load dictionary from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_ground_truth(path: str) -> Dict[str, str]:
    """Load ground truth corrections."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_cuda_benchmark(checker, typos, ground_truth, algorithm, algo_name, lang, num_words):
    """Run CUDA Batch benchmark for a specific configuration."""
    correct = 0
    total_time = 0

    # Process in batches (groups typos by length internally)
    for batch_start in range(0, len(typos), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(typos))
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

    ms_per_word = total_time / len(typos) * 1000
    accuracy = correct / len(typos) * 100

    return {
        'method': 'CUDA Batch',
        'algorithm': algo_name,
        'language': lang,
        'num_words': num_words,
        'total_time_s': total_time,
        'ms_per_word': ms_per_word,
        'correct': correct,
        'accuracy_pct': accuracy
    }


def run_c_benchmark(checker, typos, ground_truth, algorithm, algo_name, lang, num_words):
    """Run C-sequential (Numba JIT) benchmark for a specific configuration."""
    # Warmup JIT
    _ = checker.find_correction(typos[0], algorithm)

    correct = 0
    total_time = 0

    for typo in typos:
        start = time.perf_counter()
        best_word, best_dist = checker.find_correction(typo, algorithm)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if typo in ground_truth and best_word == ground_truth[typo]:
            correct += 1

    ms_per_word = total_time / len(typos) * 1000
    accuracy = correct / len(typos) * 100

    return {
        'method': 'C-Sequential',
        'algorithm': algo_name,
        'language': lang,
        'num_words': num_words,
        'total_time_s': total_time,
        'ms_per_word': ms_per_word,
        'correct': correct,
        'accuracy_pct': accuracy
    }


def main():
    print("=" * 100)
    print("CUDA BATCH vs C-SEQUENTIAL COMPREHENSIVE BENCHMARK")
    print(f"Batch size: {BATCH_SIZE} | Groups typos by length | 2D grid kernel")
    print(f"Running {NUM_ITERATIONS} iterations, reporting MEDIAN results")
    print("=" * 100)

    base = Path(__file__).parent.parent

    # Check CUDA device
    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"\nCUDA Device: {device_name}")

    # Word counts to test
    word_counts = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]

    # File suffixes for each language (MK/EN use 5000, TR uses 5K)
    word_count_files_mk_en = {
        250: '250', 500: '500', 1000: '1K', 1500: '1500',
        2000: '2K', 2500: '2500', 3500: '3500', 5000: '5000'
    }
    word_count_files_tr = {
        250: '250', 500: '500', 1000: '1K', 1500: '1500',
        2000: '2K', 2500: '2500', 3500: '3500', 5000: '5K'
    }

    # Languages
    languages = {
        'MK': {
            'dict': 'data/dictionary/mk_equal.txt',
            'gt_pattern': 'data/ground_truth/mk_hunspell_corrections_{}.json'
        },
        'EN': {
            'dict': 'data/dictionary/en_equal.txt',
            'gt_pattern': 'data/ground_truth/en_hunspell_corrections_{}.json'
        },
        'TR': {
            'dict': 'data/dictionary/tr_equal.txt',
            'gt_pattern': 'data/ground_truth/tr_hunspell_corrections_{}.json'
        }
    }

    # Algorithms
    algorithms = [
        ('levenshtein', 'Levenshtein'),
        ('damerau', 'Damerau-Levenshtein'),
        ('myers', 'Myers Bit-Vector')
    ]

    # Load dictionaries
    print("\nLoading dictionaries...")
    dictionaries = {}
    cuda_checkers = {}
    c_checkers = {}

    for lang, config in languages.items():
        dict_path = base / config['dict']
        if dict_path.exists():
            dictionaries[lang] = load_dictionary(str(dict_path))
            print(f"  {lang}: {len(dictionaries[lang]):,} words")

            # Initialize checkers
            print(f"  Initializing {lang} CUDA Batch checker...")
            cuda_checkers[lang] = CUDABatchChecker(dictionaries[lang])
            print(f"  Initializing {lang} C-baseline checker...")
            c_checkers[lang] = CBaselineChecker(dictionaries[lang])
        else:
            print(f"  WARNING: Dictionary not found for {lang}: {dict_path}")

    # Warmup CUDA Batch kernels
    print("\nWarming up CUDA Batch kernels...")
    for lang in cuda_checkers:
        gt_path = base / languages[lang]['gt_pattern'].format('250')
        if gt_path.exists():
            gt = load_ground_truth(str(gt_path))
            sample_typos = list(gt.keys())[:10]
            for algo_key, _ in algorithms:
                _ = cuda_checkers[lang].find_corrections_batch(sample_typos, algo_key)
            cp.cuda.Stream.null.synchronize()
    print("  Done")

    # Storage for all iteration results: key -> list of results
    # Key format: (method, algorithm, language, num_words)
    iteration_results = defaultdict(list)

    # Run benchmarks for NUM_ITERATIONS iterations
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'#'*100}")
        print(f"# ITERATION {iteration + 1} of {NUM_ITERATIONS}")
        print(f"{'#'*100}")

        for num_words in word_counts:
            print(f"\n{'='*80}")
            print(f"BENCHMARK: {num_words} words (Iteration {iteration + 1})")
            print(f"{'='*80}")

            for lang in ['MK', 'EN', 'TR']:
                if lang not in cuda_checkers:
                    continue

                # Determine correct file suffix for this language
                if lang == 'TR':
                    suffix = word_count_files_tr[num_words]
                else:
                    suffix = word_count_files_mk_en[num_words]

                gt_path = base / languages[lang]['gt_pattern'].format(suffix)

                if not gt_path.exists():
                    print(f"  Skipping {lang} {num_words}: ground truth not found ({gt_path})")
                    continue

                gt = load_ground_truth(str(gt_path))
                typos = list(gt.keys())
                actual_words = len(typos)

                print(f"\n--- {lang} ({actual_words} words) ---")

                for algo_key, algo_name in algorithms:
                    print(f"  {algo_name}...")
                    sys.stdout.flush()

                    # Run CUDA benchmark
                    cuda_result = run_cuda_benchmark(
                        cuda_checkers[lang], typos, gt, algo_key, algo_name, lang, actual_words
                    )
                    cuda_key = ('CUDA Batch', algo_name, lang, actual_words)
                    iteration_results[cuda_key].append(cuda_result)

                    # Run C-sequential benchmark
                    c_result = run_c_benchmark(
                        c_checkers[lang], typos, gt, algo_key, algo_name, lang, actual_words
                    )
                    c_key = ('C-Sequential', algo_name, lang, actual_words)
                    iteration_results[c_key].append(c_result)

                    # Calculate speedup for this iteration
                    speedup = c_result['ms_per_word'] / cuda_result['ms_per_word']

                    print(f"    CUDA: {cuda_result['ms_per_word']:.3f} ms/word, "
                          f"C-seq: {c_result['ms_per_word']:.2f} ms/word, "
                          f"Speedup: {speedup:.1f}x")
                    sys.stdout.flush()

    # Compute median results
    print(f"\n{'='*100}")
    print(f"COMPUTING MEDIAN RESULTS FROM {NUM_ITERATIONS} ITERATIONS")
    print(f"{'='*100}")

    all_results = []
    for key, results_list in iteration_results.items():
        method, algo_name, lang, num_words = key

        # Extract timing values
        ms_per_word_values = [r['ms_per_word'] for r in results_list]
        total_time_values = [r['total_time_s'] for r in results_list]

        # Compute statistics
        median_ms_per_word = statistics.median(ms_per_word_values)
        median_total_time = statistics.median(total_time_values)
        std_ms_per_word = statistics.stdev(ms_per_word_values) if len(ms_per_word_values) > 1 else 0.0
        min_ms_per_word = min(ms_per_word_values)
        max_ms_per_word = max(ms_per_word_values)

        # Use first result as template for non-timing fields
        median_result = {
            'method': method,
            'algorithm': algo_name,
            'language': lang,
            'num_words': num_words,
            'total_time_s': median_total_time,
            'ms_per_word': median_ms_per_word,
            'ms_per_word_median': median_ms_per_word,
            'ms_per_word_std': std_ms_per_word,
            'ms_per_word_min': min_ms_per_word,
            'ms_per_word_max': max_ms_per_word,
            'ms_per_word_all_runs': ms_per_word_values,
            'correct': results_list[0]['correct'],  # Should be same across iterations
            'accuracy_pct': results_list[0]['accuracy_pct'],  # Should be same across iterations
            'num_iterations': NUM_ITERATIONS
        }
        all_results.append(median_result)

        print(f"  {method} | {algo_name} | {lang} | {num_words} words: "
              f"median={median_ms_per_word:.4f} ms/word, std={std_ms_per_word:.4f}, "
              f"min={min_ms_per_word:.4f}, max={max_ms_per_word:.4f}")

    # Generate summary
    print("\n" + "=" * 120)
    print(f"SUMMARY: CUDA Batch vs C-Sequential Speedup (MEDIAN of {NUM_ITERATIONS} iterations)")
    print("=" * 120)

    # Create summary table
    print(f"\n{'Algorithm':<22} | {'Lang':<4} | {'Words':<6} | {'CUDA Batch (ms)':<16} | {'C-seq (ms)':<12} | {'Speedup':<10} | {'Accuracy':<10}")
    print("-" * 120)

    # Group results by configuration
    cuda_results = {(r['algorithm'], r['language'], r['num_words']): r
                    for r in all_results if r['method'] == 'CUDA Batch'}
    c_results = {(r['algorithm'], r['language'], r['num_words']): r
                 for r in all_results if r['method'] == 'C-Sequential'}

    for key in sorted(cuda_results.keys()):
        cuda_r = cuda_results[key]
        c_r = c_results.get(key)
        if c_r:
            speedup = c_r['ms_per_word'] / cuda_r['ms_per_word']
            print(f"{cuda_r['algorithm']:<22} | {cuda_r['language']:<4} | {cuda_r['num_words']:<6} | "
                  f"{cuda_r['ms_per_word']:>14.4f} | {c_r['ms_per_word']:>10.2f} | "
                  f"{speedup:>8.1f}x | {cuda_r['accuracy_pct']:>8.1f}%")

    # Save results
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        'description': f'CUDA Batch vs C-Sequential Comprehensive Benchmark (MEDIAN of {NUM_ITERATIONS} iterations)',
        'device': device_name,
        'batch_size': BATCH_SIZE,
        'num_iterations': NUM_ITERATIONS,
        'word_counts': word_counts,
        'languages': list(languages.keys()),
        'algorithms': [a[1] for a in algorithms],
        'results': all_results
    }

    output_path = results_dir / "cuda_batch_vs_c_comprehensive.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Generate CSV for easy analysis
    csv_path = results_dir / "cuda_batch_vs_c_comprehensive.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Method,Algorithm,Language,Words,Median_Time_ms,Accuracy,Num_Iterations\n")
        for r in all_results:
            f.write(f"{r['method']},{r['algorithm']},{r['language']},{r['num_words']},"
                    f"{r['ms_per_word']:.4f},{r['accuracy_pct']:.2f},{NUM_ITERATIONS}\n")

    print(f"CSV saved to: {csv_path}")

    # Generate speedup CSV
    speedup_csv_path = results_dir / "cuda_batch_speedup_vs_c.csv"
    with open(speedup_csv_path, 'w', encoding='utf-8') as f:
        f.write("Algorithm,Language,Words,CUDA_Batch_ms,C_seq_ms,Speedup,Accuracy,Num_Iterations\n")
        for key in sorted(cuda_results.keys()):
            cuda_r = cuda_results[key]
            c_r = c_results.get(key)
            if c_r:
                speedup = c_r['ms_per_word'] / cuda_r['ms_per_word']
                f.write(f"{cuda_r['algorithm']},{cuda_r['language']},{cuda_r['num_words']},"
                        f"{cuda_r['ms_per_word']:.4f},{c_r['ms_per_word']:.4f},{speedup:.2f},"
                        f"{cuda_r['accuracy_pct']:.2f},{NUM_ITERATIONS}\n")

    print(f"Speedup CSV saved to: {speedup_csv_path}")


if __name__ == "__main__":
    main()
