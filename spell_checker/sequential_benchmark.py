#!/usr/bin/env python3
"""
Sequential Baseline Benchmark for Spell Checking Algorithms - FIXED VERSION

Critical optimizations to avoid timeout:
1. Pre-groups dictionary by word length at startup
2. Runs in 3 phases automatically (10K, 50K, full+filter)
3. Tests all 6 algorithm+language combinations
4. Per-word progress reporting
5. Timeout protection per algorithm+language combo (120s default)
6. Incremental result saving after each phase

Usage:
    python spell_checker/sequential_benchmark.py
"""

import os
import sys
import json
import time
import re
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict

# Fix Windows console encoding for Cyrillic
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector


@dataclass
class PhaseResult:
    """Results from a single algorithm+language benchmark."""
    algorithm: str
    language: str
    status: str  # "OK" or "TIMEOUT"
    words_tested: int
    dict_size: int
    avg_candidates: float
    total_time_s: float
    ms_per_word: float
    correct: int
    total_with_gt: int
    accuracy_pct: float


def load_dictionary(filepath: str, max_words: Optional[int] = None) -> List[str]:
    """Load dictionary from file."""
    words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.append(w)
    if max_words and len(words) > max_words:
        words = words[:max_words]
    return words


def group_by_length(words: List[str]) -> Dict[int, List[str]]:
    """Pre-group dictionary by word length for fast candidate lookup."""
    by_length = defaultdict(list)
    for w in words:
        by_length[len(w)].append(w)
    return dict(by_length)


def get_candidates(word_len: int, dict_by_length: Dict[int, List[str]], tolerance: int = 2) -> List[str]:
    """
    Get candidate words within +/- tolerance length of the query word.
    For typo of length L, candidates = words with length L-2 to L+2.
    This reduces 819K -> ~50-100K candidates per word.
    """
    candidates = []
    for length in range(max(1, word_len - tolerance), word_len + tolerance + 1):
        if length in dict_by_length:
            candidates.extend(dict_by_length[length])
    return candidates


def load_typo_file(filepath: str) -> List[str]:
    """Load typo text file and split into words."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split on word boundaries
    words = re.findall(r'\b[\w\u0400-\u04FF]+\b', text.lower())
    return words


def find_misspelled_words(words: List[str], dict_set: Set[str], limit: int = 30) -> List[str]:
    """Find words not in dictionary (first N unique)."""
    misspelled = []
    seen = set()
    for word in words:
        if len(word) > 2 and word not in dict_set and word not in seen:
            misspelled.append(word)
            seen.add(word)
            if len(misspelled) >= limit:
                break
    return misspelled


def load_ground_truth(filepath: str) -> Dict[str, str]:
    """Load ground truth corrections from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_closest_match(
    typo: str,
    candidates: List[str],
    distance_fn: Callable[[str, str], int]
) -> Tuple[str, int]:
    """Find closest dictionary word using distance function."""
    if not candidates:
        return "", 999999

    best_match = candidates[0]
    best_dist = distance_fn(typo, candidates[0])

    for candidate in candidates[1:]:
        dist = distance_fn(typo, candidate)
        if dist < best_dist:
            best_dist = dist
            best_match = candidate
            if dist == 0:
                break

    return best_match, best_dist


class TimeoutRunner:
    """Run function with timeout using threading."""

    def __init__(self, timeout_seconds: int = 120):
        self.timeout = timeout_seconds
        self.result = None
        self.exception = None

    def run_with_timeout(self, func, *args):
        """Run function with timeout. Returns (result, timed_out)."""
        self.result = None
        self.exception = None

        def target():
            try:
                self.result = func(*args)
            except Exception as e:
                self.exception = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            return None, True  # Timed out

        if self.exception:
            raise self.exception

        return self.result, False


def run_algorithm_on_words(
    typos: List[str],
    candidates_list: List[List[str]],
    ground_truth: Dict[str, str],
    distance_fn: Callable[[str, str], int],
    algo_name: str,
    lang: str,
    timeout: int = 120
) -> PhaseResult:
    """Run algorithm on all typos with progress reporting and timeout."""

    print(f"\n--- {algo_name} + {lang} ---")

    total_time = 0.0
    total_candidates = 0
    correct = 0
    total_with_gt = 0
    words_tested = 0
    timed_out = False

    runner = TimeoutRunner(timeout_seconds=timeout)

    for i, (typo, candidates) in enumerate(zip(typos, candidates_list)):
        num_candidates = len(candidates)
        total_candidates += num_candidates

        # Time the algorithm
        start = time.perf_counter()

        try:
            result, did_timeout = runner.run_with_timeout(
                find_closest_match, typo, candidates, distance_fn
            )

            if did_timeout:
                print(f"  Word {i+1}/{len(typos)}: '{typo}' ({len(typo)} chars) -> {num_candidates:,} candidates... TIMEOUT!")
                timed_out = True
                break

            best_match, _ = result
            word_time = time.perf_counter() - start

        except Exception as e:
            print(f"  Word {i+1}/{len(typos)}: '{typo}' ERROR: {e}")
            continue

        total_time += word_time
        words_tested += 1

        # Check accuracy against ground truth
        if typo in ground_truth:
            total_with_gt += 1
            if best_match == ground_truth[typo]:
                correct += 1

        # Progress report for every word
        print(f"  Word {i+1}/{len(typos)}: '{typo}' ({len(typo)} chars) -> {num_candidates:,} candidates... {word_time:.3f}s -> '{best_match}'")

        # Check cumulative timeout
        if total_time > timeout:
            print(f"  CUMULATIVE TIMEOUT after {words_tested} words ({total_time:.1f}s > {timeout}s)")
            timed_out = True
            break

    # Calculate metrics
    if words_tested > 0:
        avg_candidates = total_candidates / words_tested
        ms_per_word = (total_time / words_tested) * 1000
        accuracy = (correct / total_with_gt * 100) if total_with_gt > 0 else 0.0
    else:
        avg_candidates = 0
        ms_per_word = 0
        accuracy = 0

    return PhaseResult(
        algorithm=algo_name,
        language=lang,
        status="TIMEOUT" if timed_out else "OK",
        words_tested=words_tested,
        dict_size=0,  # Will be set by caller
        avg_candidates=avg_candidates,
        total_time_s=total_time,
        ms_per_word=ms_per_word,
        correct=correct,
        total_with_gt=total_with_gt,
        accuracy_pct=accuracy
    )


def print_phase_summary(phase_num: int, phase_name: str, results: List[PhaseResult]):
    """Print summary table for a phase."""
    print(f"\n{'-'*100}")
    print(f"Phase {phase_num} Summary: {phase_name}")
    print(f"{'-'*100}")
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Candidates':>12} | {'Time(s)':>8} | {'ms/word':>8} | {'Accuracy':>10}")
    print(f"{'-'*22}-+-{'-'*4}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for r in results:
        if r.status == "TIMEOUT":
            print(f"{r.algorithm:<22} | {r.language:<4} | {'TIMEOUT':>12} | {'-':>8} | {'-':>8} | {'-':>10}")
        else:
            acc_str = f"{r.accuracy_pct:.1f}% ({r.correct}/{r.total_with_gt})" if r.total_with_gt > 0 else "N/A"
            print(f"{r.algorithm:<22} | {r.language:<4} | {r.avg_candidates:>12,.0f} | {r.total_time_s:>8.2f} | {r.ms_per_word:>8.1f} | {acc_str:>10}")

    print(f"{'-'*100}")


def save_results(all_phases: List[Dict], filepath: str):
    """Save results to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_phases, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filepath}")


def run_phase(
    phase_num: int,
    phase_name: str,
    typos_mk: List[str],
    typos_en: List[str],
    dict_mk: List[str],
    dict_en: List[str],
    ground_truth_mk: Dict[str, str],
    ground_truth_en: Dict[str, str],
    use_length_filter: bool,
    timeout: int = 120
) -> Dict:
    """Run a complete benchmark phase with all 6 algorithm+language combos."""

    print(f"\n{'='*80}")
    print(f"PHASE {phase_num}: {phase_name}")
    print(f"{'='*80}")

    print(f"MK dictionary size: {len(dict_mk):,}")
    print(f"EN dictionary size: {len(dict_en):,}")
    print(f"MK typos to test: {len(typos_mk)}")
    print(f"EN typos to test: {len(typos_en)}")
    print(f"Length filtering: {'YES (±2 chars)' if use_length_filter else 'NO'}")

    # Pre-group dictionaries by length
    print("\nPre-grouping dictionaries by word length...")
    dict_mk_by_len = group_by_length(dict_mk)
    dict_en_by_len = group_by_length(dict_en)

    print(f"  MK length range: {min(dict_mk_by_len.keys())}-{max(dict_mk_by_len.keys())} chars")
    print(f"  EN length range: {min(dict_en_by_len.keys())}-{max(dict_en_by_len.keys())} chars")

    # Pre-compute candidates for all typos
    print("\nPre-computing candidates for all typos...")

    if use_length_filter:
        mk_candidates = [get_candidates(len(t), dict_mk_by_len) for t in typos_mk]
        en_candidates = [get_candidates(len(t), dict_en_by_len) for t in typos_en]
    else:
        mk_candidates = [dict_mk for _ in typos_mk]
        en_candidates = [dict_en for _ in typos_en]

    if typos_mk:
        sample = typos_mk[0]
        print(f"  Sample MK: '{sample}' ({len(sample)} chars) -> {len(mk_candidates[0]):,} candidates")
    if typos_en:
        sample = typos_en[0]
        print(f"  Sample EN: '{sample}' ({len(sample)} chars) -> {len(en_candidates[0]):,} candidates")

    # Define algorithms
    algorithms = [
        ("Levenshtein", levenshtein),
        ("Damerau-Levenshtein", damerau_levenshtein),
        ("Myers Bit-Vector", myers_bitvector),
    ]

    # Define language configs
    languages = [
        ("MK", typos_mk, mk_candidates, ground_truth_mk, len(dict_mk)),
        ("EN", typos_en, en_candidates, ground_truth_en, len(dict_en)),
    ]

    results = []

    print(f"\n{'-'*80}")
    print("RUNNING BENCHMARKS (all 6 combinations)")
    print(f"{'-'*80}")

    for algo_name, algo_fn in algorithms:
        for lang, typos, candidates_list, ground_truth, dict_size in languages:
            result = run_algorithm_on_words(
                typos=typos,
                candidates_list=candidates_list,
                ground_truth=ground_truth,
                distance_fn=algo_fn,
                algo_name=algo_name,
                lang=lang,
                timeout=timeout
            )
            result.dict_size = dict_size
            results.append(result)

    # Print summary table
    print_phase_summary(phase_num, phase_name, results)

    return {
        "phase": phase_num,
        "name": phase_name,
        "use_length_filter": use_length_filter,
        "mk_dict_size": len(dict_mk),
        "en_dict_size": len(dict_en),
        "mk_typos": len(typos_mk),
        "en_typos": len(typos_en),
        "timeout": timeout,
        "results": [asdict(r) for r in results]
    }


def main():
    """Main entry point - runs all 3 phases automatically."""

    print("=" * 80)
    print("SEQUENTIAL BASELINE BENCHMARK (FIXED)")
    print("Spell Checking with Edit Distance Algorithms")
    print("=" * 80)

    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    results_file = results_dir / "sequential_baseline.json"

    # File paths
    mk_dict_path = base_dir / "data" / "dictionary" / "mk_dictionary.txt"
    en_dict_path = base_dir / "data" / "dictionary" / "en_dictionary.txt"
    mk_typo_path = base_dir / "data" / "test_texts" / "macedonian" / "mk_typos_1MB.txt"
    en_typo_path = base_dir / "data" / "test_texts" / "english" / "en_typos_1MB.txt"
    mk_gt_path = base_dir / "data" / "ground_truth" / "mk_corrections_1MB.json"
    en_gt_path = base_dir / "data" / "ground_truth" / "en_corrections_1MB.json"

    # Verify all files exist
    print("\nVerifying data files...")
    for path in [mk_dict_path, en_dict_path, mk_typo_path, en_typo_path, mk_gt_path, en_gt_path]:
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        print(f"  OK: {path.name}")

    # Load full dictionaries
    print("\nLoading dictionaries...")
    dict_mk_full = load_dictionary(str(mk_dict_path))
    dict_en_full = load_dictionary(str(en_dict_path))
    print(f"  MK dictionary: {len(dict_mk_full):,} words")
    print(f"  EN dictionary: {len(dict_en_full):,} words")

    # Load ground truth
    print("\nLoading ground truth...")
    ground_truth_mk = load_ground_truth(str(mk_gt_path))
    ground_truth_en = load_ground_truth(str(en_gt_path))
    print(f"  MK corrections: {len(ground_truth_mk):,}")
    print(f"  EN corrections: {len(ground_truth_en):,}")

    # Load typos and find misspelled words
    print("\nLoading typo files and finding misspelled words...")
    mk_words = load_typo_file(str(mk_typo_path))
    en_words = load_typo_file(str(en_typo_path))
    print(f"  MK total words: {len(mk_words):,}")
    print(f"  EN total words: {len(en_words):,}")

    dict_mk_set = set(dict_mk_full)
    dict_en_set = set(dict_en_full)

    typos_mk = find_misspelled_words(mk_words, dict_mk_set, limit=30)
    typos_en = find_misspelled_words(en_words, dict_en_set, limit=30)
    print(f"  MK misspelled found: {len(typos_mk)}")
    print(f"  EN misspelled found: {len(typos_en)}")

    if typos_mk:
        print(f"  Sample MK typos: {typos_mk[:5]}")
    if typos_en:
        print(f"  Sample EN typos: {typos_en[:5]}")

    # Check ground truth coverage
    mk_in_gt = sum(1 for t in typos_mk if t in ground_truth_mk)
    en_in_gt = sum(1 for t in typos_en if t in ground_truth_en)
    print(f"  MK typos with ground truth: {mk_in_gt}/{len(typos_mk)}")
    print(f"  EN typos with ground truth: {en_in_gt}/{len(typos_en)}")

    all_phases = []
    timeout = 120  # 120 seconds per algorithm+language combo

    # ========================================================================
    # PHASE 1: Sanity check with 10K words
    # ========================================================================
    phase1 = run_phase(
        phase_num=1,
        phase_name="30 words x 10K dict (sanity check)",
        typos_mk=typos_mk,
        typos_en=typos_en,
        dict_mk=dict_mk_full[:10000],
        dict_en=dict_en_full[:10000],
        ground_truth_mk=ground_truth_mk,
        ground_truth_en=ground_truth_en,
        use_length_filter=False,
        timeout=timeout
    )
    all_phases.append(phase1)
    save_results(all_phases, str(results_file))

    # ========================================================================
    # PHASE 2: Medium test with 50K words
    # ========================================================================
    phase2 = run_phase(
        phase_num=2,
        phase_name="30 words x 50K dict (medium test)",
        typos_mk=typos_mk,
        typos_en=typos_en,
        dict_mk=dict_mk_full[:50000],
        dict_en=dict_en_full[:50000],
        ground_truth_mk=ground_truth_mk,
        ground_truth_en=ground_truth_en,
        use_length_filter=False,
        timeout=timeout
    )
    all_phases.append(phase2)
    save_results(all_phases, str(results_file))

    # ========================================================================
    # PHASE 3: Full dictionary with length filtering
    # ========================================================================
    phase3 = run_phase(
        phase_num=3,
        phase_name="30 words x full dict + length filter (real baseline)",
        typos_mk=typos_mk,
        typos_en=typos_en,
        dict_mk=dict_mk_full,
        dict_en=dict_en_full,
        ground_truth_mk=ground_truth_mk,
        ground_truth_en=ground_truth_en,
        use_length_filter=True,
        timeout=timeout
    )
    all_phases.append(phase3)
    save_results(all_phases, str(results_file))

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE - ALL 3 PHASES")
    print("=" * 80)

    # Aggregate stats by algorithm
    print("\nAggregate Performance (across all phases):")
    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        algo_results = []
        for phase in all_phases:
            for r in phase["results"]:
                if r["algorithm"] == algo and r["status"] == "OK":
                    algo_results.append(r)

        if algo_results:
            avg_ms = sum(r["ms_per_word"] for r in algo_results) / len(algo_results)
            avg_acc = sum(r["accuracy_pct"] for r in algo_results) / len(algo_results)
            print(f"  {algo:<22}: {avg_ms:>8.2f} ms/word, {avg_acc:>5.1f}% accuracy")

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
