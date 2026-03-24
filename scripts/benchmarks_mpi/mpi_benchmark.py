#!/usr/bin/env python3
"""
MPI Parallel Benchmark for Spell Checking Algorithms

Strategy: Split CANDIDATES across processes for each typo word.
- Master loads dictionary, typos, ground truth
- For each typo: scatter candidates, compute locally, gather best results
- Master picks global best from local bests

Usage:
    mpiexec -n 1 python spell_checker/mpi_benchmark.py
    mpiexec -n 2 python spell_checker/mpi_benchmark.py
    mpiexec -n 4 python spell_checker/mpi_benchmark.py
    mpiexec -n 8 python spell_checker/mpi_benchmark.py
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set, Callable, Optional
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
    words_tested: int
    dict_size: int
    avg_candidates: float
    avg_candidates_per_proc: float
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
        # First 'remainder' processes get one extra
        size = base_size + (1 if i < remainder else 0)
        chunks.append(candidates[start:start + size])
        start += size

    return chunks


def run_mpi_benchmark(
    comm: MPI.Comm,
    typos: List[str],
    dict_by_len: Dict[int, List[str]],
    dict_size: int,
    ground_truth: Dict[str, str],
    algo_name: str,
    algo_fn: Callable,
    lang: str,
    timeout: int = 300
) -> Optional[MPIResult]:
    """Run MPI benchmark - only master returns result."""

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"\n--- [MPI-{size}] {algo_name} + {lang} ---")

    total_time = 0.0
    total_candidates = 0
    correct = 0
    with_gt = 0

    for i, typo in enumerate(typos):
        # Master gets candidates and splits them
        if rank == 0:
            all_candidates = get_candidates(len(typo), dict_by_len)
            n_cand = len(all_candidates)
            total_candidates += n_cand

            # Split candidates for distribution
            chunks = split_candidates(all_candidates, size)
            cand_per_proc = n_cand // size
        else:
            chunks = None
            n_cand = 0
            cand_per_proc = 0

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
                with_gt += 1
                if global_best_word == ground_truth[typo]:
                    correct += 1

            print(f"  Word {i+1}/{len(typos)}: '{typo}' ({len(typo)} chars) -> "
                  f"{n_cand:,} candidates ({cand_per_proc:,}/proc)... "
                  f"{elapsed:.3f}s -> '{global_best_word}'")

            if total_time > timeout:
                print(f"  TIMEOUT after {i+1} words")
                break

    # Only master returns result
    if rank == 0:
        n = len(typos)
        avg_cand = total_candidates / n if n > 0 else 0
        avg_cand_per_proc = avg_cand / size

        return MPIResult(
            algorithm=algo_name,
            language=lang,
            num_procs=size,
            words_tested=n,
            dict_size=dict_size,
            avg_candidates=avg_cand,
            avg_candidates_per_proc=avg_cand_per_proc,
            total_time_s=total_time,
            ms_per_word=total_time / n * 1000 if n > 0 else 0,
            correct=correct,
            total_with_gt=with_gt,
            accuracy_pct=(correct / with_gt * 100) if with_gt > 0 else 0
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
        print(f"MPI PARALLEL BENCHMARK ({size} processes)")
        print("=" * 80)

        # Load dictionaries
        print("\nLoading dictionaries...")
        dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
        dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
        print(f"  MK: {len(dict_mk):,} words")
        print(f"  EN: {len(dict_en):,} words")

        # Pre-group by length
        mk_by_len = group_by_length(dict_mk)
        en_by_len = group_by_length(dict_en)

        # Load ground truth
        gt_mk = load_ground_truth(str(base / "data/ground_truth/mk_corrections_1MB.json"))
        gt_en = load_ground_truth(str(base / "data/ground_truth/en_corrections_1MB.json"))

        # Find misspelled words
        mk_words = load_typos(str(base / "data/test_texts/macedonian/mk_typos_1MB.txt"))
        en_words = load_typos(str(base / "data/test_texts/english/en_typos_1MB.txt"))

        typos_mk = find_misspelled(mk_words, set(dict_mk), 50)
        typos_en = find_misspelled(en_words, set(dict_en), 50)
        print(f"  MK typos: {len(typos_mk)}")
        print(f"  EN typos: {len(typos_en)}")
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

    for algo_name, algo_fn in algorithms:
        # MK
        r = run_mpi_benchmark(
            comm, typos_mk, mk_by_len, len(dict_mk),
            gt_mk, algo_name, algo_fn, "MK", timeout=300
        )
        if r:
            results.append(r)

        # EN
        r = run_mpi_benchmark(
            comm, typos_en, en_by_len, len(dict_en),
            gt_en, algo_name, algo_fn, "EN", timeout=300
        )
        if r:
            results.append(r)

    # Master prints results and saves
    if rank == 0:
        print("\n" + "=" * 100)
        print(f"MPI BENCHMARK RESULTS ({size} processes)")
        print("=" * 100)
        print(f"{'Algorithm':<22} | {'Lang':<4} | {'Procs':>5} | {'Cand/Proc':>10} | "
              f"{'Time(s)':>8} | {'ms/word':>8} | {'Accuracy':>12}")
        print("-" * 100)

        for r in results:
            acc_str = f"{r.accuracy_pct:.1f}% ({r.correct}/{r.total_with_gt})" if r.total_with_gt else "N/A"
            print(f"{r.algorithm:<22} | {r.language:<4} | {r.num_procs:>5} | "
                  f"{r.avg_candidates_per_proc:>10,.0f} | {r.total_time_s:>8.2f} | "
                  f"{r.ms_per_word:>8.1f} | {acc_str:>12}")

        print("-" * 100)

        # Save results
        results_dir = base / "results"
        results_dir.mkdir(exist_ok=True)

        # Load existing results if any
        output_file = results_dir / "mpi_benchmark.json"
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = {"runs": []}

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


if __name__ == "__main__":
    main()
