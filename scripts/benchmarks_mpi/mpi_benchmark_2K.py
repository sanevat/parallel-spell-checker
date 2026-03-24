#!/usr/bin/env python3
"""
MPI Parallel Benchmark for Spell Checking - 2K Hunspell Version

Uses the 2000-typo Hunspell-compatible test files for benchmarking.
Splits CANDIDATES across processes for each typo word.

Usage:
    mpiexec -n 1 python spell_checker/mpi_benchmark_2K.py
    mpiexec -n 2 python spell_checker/mpi_benchmark_2K.py
    mpiexec -n 4 python spell_checker/mpi_benchmark_2K.py
    mpiexec -n 8 python spell_checker/mpi_benchmark_2K.py
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


def load_typo_text(path: str) -> List[str]:
    """Load words from typo file."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return re.findall(r'\b[\w\u0400-\u04FF]+\b', text.lower())


def load_ground_truth(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_typos(words: List[str], ground_truth: Dict[str, str]) -> List[str]:
    """Extract typos that have ground truth corrections."""
    seen = set()
    typos = []
    for w in words:
        if w in ground_truth and w not in seen:
            typos.append(w)
            seen.add(w)
    return typos


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

            # Progress every 200 words
            if (i + 1) % 200 == 0:
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
        print("=" * 80)
        print(f"MPI PARALLEL BENCHMARK - 2K Hunspell ({size} processes)")
        print("=" * 80)

        # Load dictionaries (50K each)
        print("\nLoading dictionaries...")
        dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
        dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
        print(f"  MK: {len(dict_mk):,} words")
        print(f"  EN: {len(dict_en):,} words")

        # Pre-group by length
        mk_by_len = group_by_length(dict_mk)
        en_by_len = group_by_length(dict_en)

        # Load ground truth (2K Hunspell)
        gt_mk = load_ground_truth(str(base / "data/ground_truth/mk_hunspell_corrections_2K.json"))
        gt_en = load_ground_truth(str(base / "data/ground_truth/en_hunspell_corrections_2K.json"))
        print(f"  MK ground truth: {len(gt_mk):,} entries")
        print(f"  EN ground truth: {len(gt_en):,} entries")

        # Load test texts and extract typos
        mk_words = load_typo_text(str(base / "data/test_texts/macedonian/mk_hunspell_typos_2K.txt"))
        en_words = load_typo_text(str(base / "data/test_texts/english/en_hunspell_typos_2K.txt"))

        typos_mk = extract_typos(mk_words, gt_mk)
        typos_en = extract_typos(en_words, gt_en)
        print(f"  MK typos to test: {len(typos_mk)}")
        print(f"  EN typos to test: {len(typos_en)}")
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

    if rank == 0:
        print("\n" + "=" * 80)
        print(f"RUNNING BENCHMARKS ({size} processes)")
        print("=" * 80)
        sys.stdout.flush()

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

    # Master prints results and saves
    if rank == 0:
        # Sequential baseline for speedup calculation
        seq_baseline = {
            ("Levenshtein", "MK"): 438.1,
            ("Levenshtein", "EN"): 322.2,
            ("Damerau-Levenshtein", "MK"): 478.9,
            ("Damerau-Levenshtein", "EN"): 460.0,
            ("Myers Bit-Vector", "MK"): 135.4,
            ("Myers Bit-Vector", "EN"): 120.2,
        }

        print("\n" + "=" * 110)
        print(f"MPI BENCHMARK RESULTS ({size} processes)")
        print("=" * 110)
        print(f"{'Algorithm':<22} | {'Lang':<4} | {'Procs':>5} | {'Time(s)':>8} | "
              f"{'ms/word':>8} | {'Seq ms':>8} | {'Speedup':>7} | {'Accuracy':>10}")
        print("-" * 110)

        for r in results:
            seq_ms = seq_baseline.get((r.algorithm, r.language), 0)
            speedup = seq_ms / r.ms_per_word if r.ms_per_word > 0 else 0
            acc_str = f"{r.accuracy_pct:.1f}%"
            print(f"{r.algorithm:<22} | {r.language:<4} | {r.num_procs:>5} | "
                  f"{r.total_time_s:>8.1f} | {r.ms_per_word:>8.1f} | "
                  f"{seq_ms:>8.1f} | {speedup:>7.2f}x | {acc_str:>10}")

        print("-" * 110)
        sys.stdout.flush()

        # Save results
        results_dir = base / "results"
        results_dir.mkdir(exist_ok=True)

        output_file = results_dir / "mpi_2K.json"
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = {"description": "MPI benchmark with 2K Hunspell typos, 50K dict", "runs": []}

        # Add this run
        run_data = {
            "num_procs": size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in results]
        }

        # Replace existing run with same num_procs or append
        existing_idx = next((i for i, run in enumerate(all_results["runs"])
                            if run["num_procs"] == size), None)
        if existing_idx is not None:
            all_results["runs"][existing_idx] = run_data
        else:
            all_results["runs"].append(run_data)

        # Sort by num_procs
        all_results["runs"].sort(key=lambda x: x["num_procs"])

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
