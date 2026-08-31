#!/usr/bin/env python3
"""
Run Myers Bit-Vector Sequential (1 process) Benchmark

Runs Myers bit-vector algorithm sequentially for all word counts
and updates the text_split JSON files with the results.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spell_checker.algorithms import myers_bitvector


@dataclass
class SequentialResult:
    algorithm: str
    language: str
    num_procs: int
    misspelled: int
    total_time_s: float
    ms_per_word: float
    correct: int
    accuracy_pct: float
    avg_candidates: float
    words_per_proc: float


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


def find_best_correction(typo: str, by_len: Dict[int, List[str]], dist_fn: Callable) -> Tuple[str, int]:
    candidates = get_candidates(len(typo), by_len)
    if not candidates:
        return ("", 0)
    best_word = candidates[0]
    best_dist = dist_fn(typo, candidates[0])
    for c in candidates[1:]:
        d = dist_fn(typo, c)
        if d < best_dist:
            best_dist = d
            best_word = c
            if d == 0:
                break
    return (best_word, len(candidates))


def run_sequential_benchmark(typos: List[str], by_len: Dict[int, List[str]],
                             ground_truth: Dict[str, str], lang: str) -> SequentialResult:
    """Run Myers bit-vector sequentially on all typos."""

    print(f"\n--- [SEQUENTIAL] Myers Bit-Vector + {lang} ({len(typos)} words) ---")
    sys.stdout.flush()

    corrections = {}
    total_candidates = 0

    start_time = time.perf_counter()

    for i, typo in enumerate(typos):
        best_word, num_cand = find_best_correction(typo, by_len, myers_bitvector)
        corrections[typo] = best_word
        total_candidates += num_cand

        # Progress every 100 words
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(typos)} words...")
            sys.stdout.flush()

    elapsed = time.perf_counter() - start_time

    correct = sum(1 for typo in typos if typo in ground_truth and corrections.get(typo) == ground_truth[typo])
    n = len(typos)
    ms_per_word = elapsed / n * 1000 if n > 0 else 0
    accuracy = correct / n * 100 if n > 0 else 0
    avg_cand = total_candidates / n if n > 0 else 0

    print(f"  Completed: {n} words in {elapsed:.1f}s, {correct}/{n} correct ({accuracy:.1f}%), {ms_per_word:.1f}ms/word")
    sys.stdout.flush()

    return SequentialResult(
        algorithm="Myers Bit-Vector",
        language=lang,
        num_procs=1,
        misspelled=n,
        total_time_s=elapsed,
        ms_per_word=ms_per_word,
        correct=correct,
        accuracy_pct=accuracy,
        avg_candidates=avg_cand,
        words_per_proc=float(n)
    )


def run_for_word_count(base: Path, word_count: int, suffix: str) -> List[SequentialResult]:
    """Run sequential benchmark for a specific word count."""

    print(f"\n{'='*80}")
    print(f"RUNNING SEQUENTIAL MYERS BIT-VECTOR FOR {word_count} WORDS")
    print(f"{'='*80}")

    # Load dictionaries
    print("\nLoading dictionaries...")
    dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
    print(f"  MK: {len(dict_mk):,} words, EN: {len(dict_en):,} words")

    mk_by_len = group_by_length(dict_mk)
    en_by_len = group_by_length(dict_en)

    # Load ground truth
    gt_mk_path = base / f"data/ground_truth/mk_hunspell_corrections_{suffix}.json"
    gt_en_path = base / f"data/ground_truth/en_hunspell_corrections_{suffix}.json"

    with open(gt_mk_path, 'r', encoding='utf-8') as f:
        gt_mk = json.load(f)
    with open(gt_en_path, 'r', encoding='utf-8') as f:
        gt_en = json.load(f)

    typos_mk = list(gt_mk.keys())
    typos_en = list(gt_en.keys())
    print(f"  MK typos: {len(typos_mk)}, EN typos: {len(typos_en)}")
    sys.stdout.flush()

    results = []

    # Run for MK
    result_mk = run_sequential_benchmark(typos_mk, mk_by_len, gt_mk, "MK")
    results.append(result_mk)

    # Run for EN
    result_en = run_sequential_benchmark(typos_en, en_by_len, gt_en, "EN")
    results.append(result_en)

    return results


def update_json_file(json_path: Path, results: List[SequentialResult]):
    """Update the JSON file with sequential results."""

    # Load existing data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Add "1" key for 1 process results
    if 'text_split' not in data:
        data['text_split'] = {}

    data['text_split']['1'] = [asdict(r) for r in results]

    # Save updated data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Updated: {json_path}")


def main():
    print("=" * 80)
    print("MYERS BIT-VECTOR SEQUENTIAL BENCHMARK")
    print("Adding 1-process results to all text_split JSON files")
    print("=" * 80)

    base = Path(__file__).parent.parent
    results_dir = base / "results"

    # Define word counts and their file suffixes
    word_configs = [
        (250, "250", "text_split_250.json"),
        (500, "500", "text_split_500.json"),
        (1000, "1K", "text_split_1K.json"),
        (1500, "1500", "text_split_1500.json"),
        (2000, "2K", "text_split_2K.json"),
        (2500, "2500", "text_split_2500.json"),
        (3500, "3500", "text_split_3500.json"),
        (5000, "5000", "text_split_5000.json"),
    ]

    all_results = {}

    for word_count, suffix, json_file in word_configs:
        json_path = results_dir / json_file

        if not json_path.exists():
            print(f"\nSkipping {json_file} - file not found")
            continue

        # Run benchmark
        results = run_for_word_count(base, word_count, suffix)
        all_results[word_count] = results

        # Update JSON file
        update_json_file(json_path, results)

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - MYERS BIT-VECTOR SEQUENTIAL (1 process)")
    print("=" * 80)
    print(f"{'Words':<8} {'MK Time(s)':>12} {'MK ms/word':>12} {'EN Time(s)':>12} {'EN ms/word':>12}")
    print("-" * 60)

    for word_count in sorted(all_results.keys()):
        results = all_results[word_count]
        mk_result = next((r for r in results if r.language == "MK"), None)
        en_result = next((r for r in results if r.language == "EN"), None)

        mk_time = mk_result.total_time_s if mk_result else 0
        mk_ms = mk_result.ms_per_word if mk_result else 0
        en_time = en_result.total_time_s if en_result else 0
        en_ms = en_result.ms_per_word if en_result else 0

        print(f"{word_count:<8} {mk_time:>12.2f} {mk_ms:>12.2f} {en_time:>12.2f} {en_ms:>12.2f}")

    print("=" * 80)
    print("Done! All JSON files updated with 1-process Myers Bit-Vector results.")


if __name__ == "__main__":
    main()
