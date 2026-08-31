#!/usr/bin/env python3
"""
Fix TR 5K accuracy anomaly in cuda_batch_vs_c_comprehensive.json

Runs CUDA Batch vs C-Sequential benchmark for Turkish 5K only (4 iterations)
and updates the JSON file with corrected results.

Usage:
    python scripts/fix_tr_5k.py
"""

import os
import sys
import json
import time
import statistics
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spell_checker.cuda_batch import CUDABatchChecker
from spell_checker.c_baseline import CBaselineChecker
import cupy as cp

BATCH_SIZE = 64
NUM_ITERATIONS = 4

base = Path(__file__).parent.parent


def load_dictionary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_ground_truth(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print('=' * 70)
    print('FIX TR 5K ACCURACY ANOMALY')
    print('CUDA Batch vs C-Sequential Benchmark - Turkish 5K Only')
    print(f'Iterations: {NUM_ITERATIONS}, Batch size: {BATCH_SIZE}')
    print('=' * 70)

    # Load dictionary
    print('\nLoading TR dictionary...')
    dict_tr = load_dictionary(base / 'data/dictionary/tr_equal.txt')
    print(f'  Loaded: {len(dict_tr):,} words')

    # Load ground truth
    print('Loading TR 5K ground truth...')
    gt = load_ground_truth(base / 'data/ground_truth/tr_hunspell_corrections_5K.json')
    typos = list(gt.keys())
    print(f'  Loaded: {len(typos)} typos')

    # Verify corrections exist in dictionary
    dict_set = set(dict_tr)
    coverage = sum(1 for c in gt.values() if c.lower() in dict_set)
    print(f'  Coverage: {coverage}/{len(gt)} = {coverage/len(gt)*100:.1f}%')

    # Initialize checkers
    print('\nInitializing checkers...')
    cuda_checker = CUDABatchChecker(dict_tr)
    c_checker = CBaselineChecker(dict_tr)
    print('  Done')

    # Warmup CUDA kernels
    print('Warming up CUDA kernels...')
    for algo in ['levenshtein', 'damerau', 'myers']:
        _ = cuda_checker.find_corrections_batch(typos[:10], algo)
    cp.cuda.Stream.null.synchronize()
    print('  Done')

    algorithms = [
        ('levenshtein', 'Levenshtein'),
        ('damerau', 'Damerau-Levenshtein'),
        ('myers', 'Myers Bit-Vector')
    ]

    results = []

    for algo_key, algo_name in algorithms:
        print(f'\n{"="*70}')
        print(f'Algorithm: {algo_name}')
        print('=' * 70)

        cuda_runs = []
        c_runs = []
        cuda_correct = 0
        c_correct = 0

        for iteration in range(NUM_ITERATIONS):
            print(f'\n  Iteration {iteration + 1}/{NUM_ITERATIONS}')

            # CUDA Batch benchmark
            correct = 0
            total_time = 0
            for batch_start in range(0, len(typos), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(typos))
                batch_typos = typos[batch_start:batch_end]

                start = time.perf_counter()
                batch_results = cuda_checker.find_corrections_batch(batch_typos, algo_key)
                cp.cuda.Stream.null.synchronize()
                elapsed = time.perf_counter() - start
                total_time += elapsed

                for i, (best_word, best_dist) in enumerate(batch_results):
                    typo = batch_typos[i]
                    if typo in gt and best_word == gt[typo]:
                        correct += 1

            ms_per_word = total_time / len(typos) * 1000
            cuda_runs.append(ms_per_word)
            if iteration == 0:
                cuda_correct = correct
            print(f'    CUDA Batch: {ms_per_word:.4f} ms/word, {correct}/{len(typos)} correct')

            # C Sequential benchmark
            if iteration == 0:  # Warmup JIT
                _ = c_checker.find_correction(typos[0], algo_key)

            correct = 0
            total_time = 0
            for typo in typos:
                start = time.perf_counter()
                best_word, best_dist = c_checker.find_correction(typo, algo_key)
                elapsed = time.perf_counter() - start
                total_time += elapsed

                if typo in gt and best_word == gt[typo]:
                    correct += 1

            ms_per_word = total_time / len(typos) * 1000
            c_runs.append(ms_per_word)
            if iteration == 0:
                c_correct = correct
            print(f'    C-Sequential: {ms_per_word:.4f} ms/word, {correct}/{len(typos)} correct')

        # Store CUDA result
        results.append({
            'method': 'CUDA Batch',
            'algorithm': algo_name,
            'language': 'TR',
            'num_words': len(typos),
            'total_time_s': statistics.median(cuda_runs) * len(typos) / 1000,
            'ms_per_word': statistics.median(cuda_runs),
            'ms_per_word_median': statistics.median(cuda_runs),
            'ms_per_word_std': statistics.stdev(cuda_runs) if len(cuda_runs) > 1 else 0,
            'ms_per_word_min': min(cuda_runs),
            'ms_per_word_max': max(cuda_runs),
            'ms_per_word_all_runs': cuda_runs,
            'correct': cuda_correct,
            'accuracy_pct': cuda_correct / len(typos) * 100,
            'num_iterations': NUM_ITERATIONS
        })

        # Store C result
        results.append({
            'method': 'C-Sequential',
            'algorithm': algo_name,
            'language': 'TR',
            'num_words': len(typos),
            'total_time_s': statistics.median(c_runs) * len(typos) / 1000,
            'ms_per_word': statistics.median(c_runs),
            'ms_per_word_median': statistics.median(c_runs),
            'ms_per_word_std': statistics.stdev(c_runs) if len(c_runs) > 1 else 0,
            'ms_per_word_min': min(c_runs),
            'ms_per_word_max': max(c_runs),
            'ms_per_word_all_runs': c_runs,
            'correct': c_correct,
            'accuracy_pct': c_correct / len(typos) * 100,
            'num_iterations': NUM_ITERATIONS
        })

        print(f'\n  Summary:')
        print(f'    CUDA Batch:   {statistics.median(cuda_runs):.4f} ms/word (median), {cuda_correct/len(typos)*100:.1f}% accuracy')
        print(f'    C-Sequential: {statistics.median(c_runs):.4f} ms/word (median), {c_correct/len(typos)*100:.1f}% accuracy')
        print(f'    Speedup:      {statistics.median(c_runs)/statistics.median(cuda_runs):.2f}x')

    # Update JSON file
    print('\n' + '=' * 70)
    print('UPDATING JSON FILE')
    print('=' * 70)

    json_file = base / 'results/cuda_batch_vs_c_comprehensive.json'

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find and replace TR 5000/4998 entries
    updated = 0
    for i, entry in enumerate(data['results']):
        if entry['language'] == 'TR' and entry['num_words'] in [4998, 5000]:
            # Find matching new result
            for new_result in results:
                if new_result['method'] == entry['method'] and new_result['algorithm'] == entry['algorithm']:
                    old_acc = entry['accuracy_pct']
                    new_acc = new_result['accuracy_pct']
                    print(f"  {entry['method']} {entry['algorithm']}: {old_acc:.1f}% -> {new_acc:.1f}%")
                    data['results'][i] = new_result
                    updated += 1
                    break

    print(f'\nUpdated {updated} entries')

    # Save
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f'Saved: {json_file}')

    # Print final summary
    print('\n' + '=' * 70)
    print('FINAL RESULTS - TR 5K')
    print('=' * 70)
    print(f'{"Method":<15} {"Algorithm":<20} {"ms/word":>10} {"Accuracy":>10}')
    print('-' * 60)
    for r in results:
        print(f"{r['method']:<15} {r['algorithm']:<20} {r['ms_per_word']:>10.4f} {r['accuracy_pct']:>9.1f}%")

    print('\nDONE!')


if __name__ == '__main__':
    main()
