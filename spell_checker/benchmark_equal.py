#!/usr/bin/env python3
"""
Sequential Baseline Benchmark with Equal-Size Hunspell Dictionaries

Tests UTF-8/Cyrillic overhead by using identical dictionary sizes (50K each).
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector


@dataclass
class Result:
    algorithm: str
    language: str
    words_tested: int
    dict_size: int
    avg_candidates: float
    total_time_s: float
    ms_per_word: float
    correct: int
    total_with_gt: int
    accuracy_pct: float
    status: str = "OK"


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


def load_typos(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return re.findall(r'\b[\w\u0400-\u04FF]+\b', text.lower())


def find_misspelled(words: List[str], dict_set: Set[str], limit: int) -> List[str]:
    result = []
    seen = set()
    for w in words:
        if len(w) > 2 and w not in dict_set and w not in seen:
            result.append(w)
            seen.add(w)
            if len(result) >= limit:
                break
    return result


def load_ground_truth(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_best_match(typo: str, candidates: List[str], dist_fn) -> Tuple[str, int]:
    if not candidates:
        return "", 999999
    best = candidates[0]
    best_d = dist_fn(typo, candidates[0])
    for c in candidates[1:]:
        d = dist_fn(typo, c)
        if d < best_d:
            best_d = d
            best = c
            if d == 0:
                break
    return best, best_d


def run_benchmark(
    typos: List[str],
    dict_by_len: Dict[int, List[str]],
    dict_size: int,
    ground_truth: Dict[str, str],
    algo_name: str,
    algo_fn: Callable,
    lang: str,
    timeout: int = 300
) -> Result:
    print(f"\n--- {algo_name} + {lang} ---")

    total_time = 0.0
    total_cand = 0
    correct = 0
    with_gt = 0

    for i, typo in enumerate(typos):
        candidates = get_candidates(len(typo), dict_by_len)
        n_cand = len(candidates)
        total_cand += n_cand

        start = time.perf_counter()
        best, _ = find_best_match(typo, candidates, algo_fn)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if typo in ground_truth:
            with_gt += 1
            if best == ground_truth[typo]:
                correct += 1

        print(f"  Word {i+1}/{len(typos)}: '{typo}' ({len(typo)} chars) -> {n_cand:,} candidates... {elapsed:.3f}s -> '{best}'")

        if total_time > timeout:
            print(f"  TIMEOUT after {i+1} words")
            return Result(
                algorithm=algo_name, language=lang, words_tested=i+1,
                dict_size=dict_size, avg_candidates=total_cand/(i+1),
                total_time_s=total_time, ms_per_word=total_time/(i+1)*1000,
                correct=correct, total_with_gt=with_gt,
                accuracy_pct=(correct/with_gt*100) if with_gt else 0,
                status="TIMEOUT"
            )

    n = len(typos)
    return Result(
        algorithm=algo_name, language=lang, words_tested=n,
        dict_size=dict_size, avg_candidates=total_cand/n,
        total_time_s=total_time, ms_per_word=total_time/n*1000,
        correct=correct, total_with_gt=with_gt,
        accuracy_pct=(correct/with_gt*100) if with_gt else 0
    )


def main():
    print("=" * 80)
    print("SEQUENTIAL BASELINE - EQUAL HUNSPELL DICTIONARIES (50K each)")
    print("=" * 80)

    base = Path(__file__).parent.parent

    # Load equal dictionaries
    print("\nLoading equal-size dictionaries (50K each)...")
    dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
    print(f"  MK: {len(dict_mk):,} words")
    print(f"  EN: {len(dict_en):,} words")

    # Pre-group by length
    print("\nPre-grouping by word length...")
    mk_by_len = group_by_length(dict_mk)
    en_by_len = group_by_length(dict_en)
    print(f"  MK length range: {min(mk_by_len.keys())}-{max(mk_by_len.keys())}")
    print(f"  EN length range: {min(en_by_len.keys())}-{max(en_by_len.keys())}")

    # Load ground truth
    print("\nLoading ground truth...")
    gt_mk = load_ground_truth(str(base / "data/ground_truth/mk_corrections_1MB.json"))
    gt_en = load_ground_truth(str(base / "data/ground_truth/en_corrections_1MB.json"))
    print(f"  MK: {len(gt_mk):,} corrections")
    print(f"  EN: {len(gt_en):,} corrections")

    # Find misspelled words
    print("\nFinding misspelled words (50 per language)...")
    mk_words = load_typos(str(base / "data/test_texts/macedonian/mk_typos_1MB.txt"))
    en_words = load_typos(str(base / "data/test_texts/english/en_typos_1MB.txt"))

    typos_mk = find_misspelled(mk_words, set(dict_mk), 50)
    typos_en = find_misspelled(en_words, set(dict_en), 50)
    print(f"  MK typos: {len(typos_mk)}")
    print(f"  EN typos: {len(typos_en)}")
    print(f"  Sample MK: {typos_mk[:5]}")
    print(f"  Sample EN: {typos_en[:5]}")

    # Check ground truth coverage
    mk_in_gt = sum(1 for t in typos_mk if t in gt_mk)
    en_in_gt = sum(1 for t in typos_en if t in gt_en)
    print(f"  MK with ground truth: {mk_in_gt}/{len(typos_mk)}")
    print(f"  EN with ground truth: {en_in_gt}/{len(typos_en)}")

    # Algorithms
    algorithms = [
        ("Levenshtein", levenshtein),
        ("Damerau-Levenshtein", damerau_levenshtein),
        ("Myers Bit-Vector", myers_bitvector),
    ]

    results = []

    print("\n" + "=" * 80)
    print("RUNNING BENCHMARKS (50 words x 50K dict, length filter ±2)")
    print("=" * 80)

    for algo_name, algo_fn in algorithms:
        # MK
        r_mk = run_benchmark(typos_mk, mk_by_len, len(dict_mk), gt_mk, algo_name, algo_fn, "MK", timeout=300)
        results.append(r_mk)

        # EN
        r_en = run_benchmark(typos_en, en_by_len, len(dict_en), gt_en, algo_name, algo_fn, "EN", timeout=300)
        results.append(r_en)

    # Print summary table
    print("\n" + "=" * 100)
    print("SEQUENTIAL BASELINE (Hunspell 50K equal dictionaries)")
    print("=" * 100)
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Candidates':>12} | {'Time(s)':>8} | {'ms/word':>8} | {'Accuracy':>12}")
    print("-" * 100)

    for r in results:
        acc_str = f"{r.accuracy_pct:.1f}% ({r.correct}/{r.total_with_gt})" if r.total_with_gt else "N/A"
        status = " [TIMEOUT]" if r.status == "TIMEOUT" else ""
        print(f"{r.algorithm:<22} | {r.language:<4} | {r.avg_candidates:>12,.0f} | {r.total_time_s:>8.2f} | {r.ms_per_word:>8.1f} | {acc_str:>12}{status}")

    print("-" * 100)

    # UTF-8 overhead analysis
    print("\n" + "=" * 60)
    print("UTF-8 OVERHEAD ANALYSIS (MK Cyrillic vs EN ASCII)")
    print("=" * 60)

    for algo_name in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        mk_res = next((r for r in results if r.algorithm == algo_name and r.language == "MK"), None)
        en_res = next((r for r in results if r.algorithm == algo_name and r.language == "EN"), None)

        if mk_res and en_res and mk_res.status == "OK" and en_res.status == "OK":
            overhead = (mk_res.ms_per_word / en_res.ms_per_word - 1) * 100
            if overhead > 0:
                print(f"{algo_name:<22}: MK is {overhead:+.1f}% SLOWER than EN")
            else:
                print(f"{algo_name:<22}: MK is {abs(overhead):.1f}% FASTER than EN")
        else:
            print(f"{algo_name:<22}: Cannot compare (timeout or error)")

    # Save results
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "sequential_baseline_fair.json"

    output = {
        "description": "Sequential baseline with equal-size Hunspell dictionaries (50K each)",
        "mk_dict_size": len(dict_mk),
        "en_dict_size": len(dict_en),
        "mk_typos": len(typos_mk),
        "en_typos": len(typos_en),
        "length_filter": "±2",
        "results": [asdict(r) for r in results]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
