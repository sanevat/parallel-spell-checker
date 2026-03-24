#!/usr/bin/env python3
"""
MPI Text-Split Spell Checker (Petrushevski approach)

Strategy: Split WORDS across processes (not dictionary candidates).
- Master splits typos into N equal chunks
- Each process gets its chunk + FULL dictionary
- Each process independently finds corrections for its chunk
- Master gathers all results at the end
- Communication: scatter/gather ONCE (not per word)

Usage:
    mpiexec -n 2 python spell_checker/mpi_text_split.py
    mpiexec -n 4 python spell_checker/mpi_text_split.py
    mpiexec -n 8 python spell_checker/mpi_text_split.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# MPI import
try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py not installed. Run: pip install mpi4py")
    sys.exit(1)

# Add parent to path for algorithm imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector


@dataclass
class TextSplitResult:
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


def load_ground_truth(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_best_correction(typo: str, by_len: Dict[int, List[str]], dist_fn: Callable) -> Tuple[str, int]:
    """Find best correction for a single typo using length-filtered candidates."""
    candidates = get_candidates(len(typo), by_len)

    if not candidates:
        return ("", 999999)

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


def process_chunk(typos: List[str], by_len: Dict[int, List[str]], dist_fn: Callable) -> List[Tuple[str, str, int]]:
    """Process a chunk of typos and return list of (typo, correction, num_candidates)."""
    results = []
    for typo in typos:
        best_word, num_cand = find_best_correction(typo, by_len, dist_fn)
        results.append((typo, best_word, num_cand))
    return results


def split_list(lst: List, n: int) -> List[List]:
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def run_text_split_benchmark(
    comm: MPI.Comm,
    typos: List[str],
    by_len: Dict[int, List[str]],
    ground_truth: Dict[str, str],
    algo_name: str,
    algo_fn: Callable,
    lang: str,
) -> TextSplitResult:
    """Run text-split MPI benchmark."""

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only master prints
    if rank == 0:
        print(f"\n--- [TEXT-SPLIT MPI-{size}] {algo_name} + {lang} ({len(typos)} words) ---")
        sys.stdout.flush()

    # Master splits typos into chunks
    if rank == 0:
        chunks = split_list(typos, size)
        words_per_proc = len(typos) / size
    else:
        chunks = None
        words_per_proc = 0

    # Broadcast dictionary to all (they all need it)
    by_len = comm.bcast(by_len, root=0)

    # Scatter typo chunks to all processes
    local_typos = comm.scatter(chunks, root=0)

    # Synchronize before timing
    comm.Barrier()
    start_time = time.perf_counter()

    # Each process works on its chunk independently
    local_results = process_chunk(local_typos, by_len, algo_fn)

    # Synchronize after computation
    comm.Barrier()
    elapsed = time.perf_counter() - start_time

    # Gather all results to master
    all_results = comm.gather(local_results, root=0)

    # Master calculates accuracy
    if rank == 0:
        # Flatten results
        corrections = {}
        total_candidates = 0
        for proc_results in all_results:
            for typo, correction, num_cand in proc_results:
                corrections[typo] = correction
                total_candidates += num_cand

        # Calculate accuracy
        correct = 0
        for typo in typos:
            if typo in ground_truth and corrections.get(typo) == ground_truth[typo]:
                correct += 1

        n = len(typos)
        ms_per_word = elapsed / n * 1000
        accuracy = correct / n * 100
        avg_cand = total_candidates / n

        print(f"  Completed: {n} words in {elapsed:.1f}s, "
              f"{correct}/{n} correct ({accuracy:.1f}%), {ms_per_word:.1f}ms/word")
        sys.stdout.flush()

        return TextSplitResult(
            algorithm=algo_name,
            language=lang,
            num_procs=size,
            misspelled=n,
            total_time_s=elapsed,
            ms_per_word=ms_per_word,
            correct=correct,
            accuracy_pct=accuracy,
            avg_candidates=avg_cand,
            words_per_proc=words_per_proc
        )

    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    base = Path(__file__).parent.parent

    # Only master loads data
    if rank == 0:
        print("=" * 80)
        print(f"TEXT-SPLIT MPI BENCHMARK ({size} processes)")
        print("=" * 80)

        # Load dictionaries
        print("\nLoading dictionaries...")
        dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
        dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
        print(f"  MK: {len(dict_mk):,} words, EN: {len(dict_en):,} words")

        # Pre-group by length
        mk_by_len = group_by_length(dict_mk)
        en_by_len = group_by_length(dict_en)

        # Load ground truth (500 word version)
        gt_mk = load_ground_truth(str(base / "data/ground_truth/mk_hunspell_corrections_500.json"))
        gt_en = load_ground_truth(str(base / "data/ground_truth/en_hunspell_corrections_500.json"))

        # Extract typos
        typos_mk = list(gt_mk.keys())
        typos_en = list(gt_en.keys())

        print(f"  MK typos: {len(typos_mk)}, EN typos: {len(typos_en)}")
        print(f"  Words per process: {len(typos_mk) // size} MK, {len(typos_en) // size} EN")
        sys.stdout.flush()
    else:
        mk_by_len = en_by_len = None
        gt_mk = gt_en = None
        typos_mk = typos_en = None

    # Broadcast ground truth and typos to all
    gt_mk = comm.bcast(gt_mk, root=0)
    gt_en = comm.bcast(gt_en, root=0)
    typos_mk = comm.bcast(typos_mk, root=0)
    typos_en = comm.bcast(typos_en, root=0)

    # Algorithms
    algorithms = [
        ("Levenshtein", levenshtein),
        ("Damerau-Levenshtein", damerau_levenshtein),
        ("Myers Bit-Vector", myers_bitvector),
    ]

    results = []

    for algo_name, algo_fn in algorithms:
        # MK
        r = run_text_split_benchmark(
            comm, typos_mk, mk_by_len, gt_mk, algo_name, algo_fn, "MK"
        )
        if r:
            results.append(r)

        # EN
        r = run_text_split_benchmark(
            comm, typos_en, en_by_len, gt_en, algo_name, algo_fn, "EN"
        )
        if r:
            results.append(r)

    # Master saves results
    if rank == 0:
        print(f"\n--- TEXT-SPLIT MPI-{size} Summary ---")
        for r in results:
            print(f"  {r.algorithm} {r.language}: {r.ms_per_word:.1f}ms/word, {r.accuracy_pct:.1f}%")

        # Save to temp file
        results_dir = base / "results"
        results_dir.mkdir(exist_ok=True)

        output_file = results_dir / "text_split_temp.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "num_procs": size,
                "results": [asdict(r) for r in results]
            }, f, indent=2, ensure_ascii=False)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
