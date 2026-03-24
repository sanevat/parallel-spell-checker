#!/usr/bin/env python3
"""
MPI Parallel Benchmark for Spell Checking - 500 word version.
Progress reported every 50 words.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional
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
class MPIResult:
    algorithm: str
    language: str
    num_procs: int
    misspelled: int
    total_time_s: float
    ms_per_word: float
    correct: int
    accuracy_pct: float
    avg_candidates: float
    avg_candidates_per_proc: float


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


def find_local_best(typo: str, candidates: List[str], dist_fn: Callable) -> Tuple[str, int]:
    """Find best match in local candidate list."""
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

    return (best_word, best_dist)


def split_candidates(candidates: List[str], num_procs: int) -> List[List[str]]:
    """Split candidates evenly across processes."""
    n = len(candidates)
    base_size = n // num_procs
    remainder = n % num_procs

    chunks = []
    start = 0
    for i in range(num_procs):
        size = base_size + (1 if i < remainder else 0)
        chunks.append(candidates[start:start + size])
        start += size

    return chunks


def run_mpi_benchmark(
    comm: MPI.Comm,
    typos: List[str],
    dict_by_len: Dict[int, List[str]],
    ground_truth: Dict[str, str],
    algo_name: str,
    algo_fn: Callable,
    lang: str,
) -> Optional[MPIResult]:
    """Run MPI benchmark - only master returns result."""

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"\n--- [MPI-{size}] {algo_name} + {lang} ({len(typos)} words) ---")
        sys.stdout.flush()

    total_time = 0.0
    total_candidates = 0
    correct = 0

    for i, typo in enumerate(typos):
        # Master gets candidates and splits them
        if rank == 0:
            all_candidates = get_candidates(len(typo), dict_by_len)
            n_cand = len(all_candidates)
            total_candidates += n_cand
            chunks = split_candidates(all_candidates, size)
        else:
            chunks = None
            n_cand = 0

        # Broadcast typo to all processes
        typo = comm.bcast(typo, root=0)

        # Scatter candidate chunks
        local_candidates = comm.scatter(chunks, root=0)

        # Synchronize before timing
        comm.Barrier()
        start = time.perf_counter()

        # Each process finds its local best
        local_best_word, local_best_dist = find_local_best(typo, local_candidates, algo_fn)

        # Gather all local bests to master
        all_bests = comm.gather((local_best_word, local_best_dist), root=0)

        # Synchronize after computation
        comm.Barrier()
        elapsed = time.perf_counter() - start

        # Master picks global best
        if rank == 0:
            total_time += elapsed

            # Find global best from all local bests
            global_best_word = ""
            global_best_dist = 999999
            for word, dist in all_bests:
                if dist < global_best_dist:
                    global_best_dist = dist
                    global_best_word = word

            # Check accuracy
            if typo in ground_truth:
                if global_best_word == ground_truth[typo]:
                    correct += 1

            # Progress every 50 words
            if (i + 1) % 50 == 0:
                elapsed_total = total_time
                avg_ms = elapsed_total / (i + 1) * 1000
                print(f"  Progress: {i+1}/{len(typos)} words, "
                      f"elapsed: {elapsed_total:.1f}s, avg: {avg_ms:.1f}ms/word")
                sys.stdout.flush()

    # Only master returns result
    if rank == 0:
        n = len(typos)
        avg_cand = total_candidates / n if n > 0 else 0
        avg_cand_per_proc = avg_cand / size

        print(f"  Completed: {n} words in {total_time:.1f}s, "
              f"{correct}/{n} correct ({correct/n*100:.1f}%)")
        sys.stdout.flush()

        return MPIResult(
            algorithm=algo_name,
            language=lang,
            num_procs=size,
            misspelled=n,
            total_time_s=total_time,
            ms_per_word=total_time / n * 1000 if n > 0 else 0,
            correct=correct,
            accuracy_pct=(correct / n * 100) if n > 0 else 0,
            avg_candidates=avg_cand,
            avg_candidates_per_proc=avg_cand_per_proc
        )

    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    base = Path(__file__).parent.parent

    # Only master loads data
    if rank == 0:
        print(f"\n--- MPI-{size} Benchmark (500 words) ---")

        # Load dictionaries (50K each)
        dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
        dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))

        # Pre-group by length
        mk_by_len = group_by_length(dict_mk)
        en_by_len = group_by_length(dict_en)

        # Load ground truth (500 word version)
        gt_mk = load_ground_truth(str(base / "data/ground_truth/mk_hunspell_corrections_500.json"))
        gt_en = load_ground_truth(str(base / "data/ground_truth/en_hunspell_corrections_500.json"))

        # Extract typos from ground truth
        typos_mk = list(gt_mk.keys())
        typos_en = list(gt_en.keys())

        print(f"  MK typos: {len(typos_mk)}, EN typos: {len(typos_en)}")
        sys.stdout.flush()
    else:
        dict_mk = dict_en = None
        mk_by_len = en_by_len = None
        gt_mk = gt_en = None
        typos_mk = typos_en = None

    # Broadcast data to all processes
    dict_mk = comm.bcast(dict_mk, root=0)
    dict_en = comm.bcast(dict_en, root=0)
    mk_by_len = comm.bcast(mk_by_len, root=0)
    en_by_len = comm.bcast(en_by_len, root=0)
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
        r = run_mpi_benchmark(
            comm, typos_mk, mk_by_len, gt_mk, algo_name, algo_fn, "MK"
        )
        if r:
            results.append(r)

        # EN
        r = run_mpi_benchmark(
            comm, typos_en, en_by_len, gt_en, algo_name, algo_fn, "EN"
        )
        if r:
            results.append(r)

    # Master saves results
    if rank == 0:
        print(f"\n--- MPI-{size} Results ---")
        for r in results:
            print(f"  {r.algorithm} {r.language}: {r.ms_per_word:.1f}ms/word, {r.accuracy_pct:.1f}%")

        # Save to temp file for main script to read
        results_dir = base / "results"
        results_dir.mkdir(exist_ok=True)

        output_file = results_dir / "mpi_500_temp.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "num_procs": size,
                "results": [asdict(r) for r in results]
            }, f, indent=2, ensure_ascii=False)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
