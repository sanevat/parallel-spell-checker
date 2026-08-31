#!/usr/bin/env python3
"""
MPI Dict-Split Spell Checker for Turkish - Damerau-Levenshtein.
Strategy: Split CANDIDATES across processes for each typo word.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py not installed")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spell_checker.algorithms import damerau_levenshtein


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


def find_local_best(typo: str, candidates: List[str]) -> Tuple[str, int]:
    if not candidates:
        return ("", 999999)

    best_word = candidates[0]
    best_dist = damerau_levenshtein(typo, candidates[0])

    for c in candidates[1:]:
        d = damerau_levenshtein(typo, c)
        if d < best_dist:
            best_dist = d
            best_word = c
            if d == 0:
                break

    return (best_word, best_dist)


def split_candidates(candidates: List[str], num_procs: int) -> List[List[str]]:
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


def run_benchmark(comm, typos, dict_by_len, ground_truth):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"\n--- [DICT-SPLIT MPI-{size}] Damerau-Levenshtein + TR ({len(typos)} words) ---")
        sys.stdout.flush()

    total_time = 0.0
    total_candidates = 0
    correct = 0

    for i, typo in enumerate(typos):
        if rank == 0:
            all_candidates = get_candidates(len(typo), dict_by_len)
            n_cand = len(all_candidates)
            total_candidates += n_cand
            chunks = split_candidates(all_candidates, size)
        else:
            chunks = None
            n_cand = 0

        typo = comm.bcast(typo, root=0)
        local_candidates = comm.scatter(chunks, root=0)

        comm.Barrier()
        start = time.perf_counter()

        local_best_word, local_best_dist = find_local_best(typo, local_candidates)

        all_bests = comm.gather((local_best_word, local_best_dist), root=0)

        comm.Barrier()
        elapsed = time.perf_counter() - start

        if rank == 0:
            total_time += elapsed

            global_best_word = ""
            global_best_dist = 999999
            for word, dist in all_bests:
                if dist < global_best_dist:
                    global_best_dist = dist
                    global_best_word = word

            if typo in ground_truth:
                if global_best_word == ground_truth[typo]:
                    correct += 1

            if (i + 1) % 50 == 0:
                avg_ms = total_time / (i + 1) * 1000
                print(f"  Progress: {i+1}/{len(typos)} words, elapsed: {total_time:.1f}s, avg: {avg_ms:.1f}ms/word")
                sys.stdout.flush()

    if rank == 0:
        n = len(typos)
        avg_cand = total_candidates / n if n > 0 else 0

        print(f"  Completed: {n} words in {total_time:.1f}s, {correct}/{n} correct ({correct/n*100:.1f}%)")
        sys.stdout.flush()

        return {
            'algorithm': 'Damerau-Levenshtein',
            'language': 'TR',
            'num_procs': size,
            'misspelled': n,
            'total_time_s': total_time,
            'ms_per_word': total_time / n * 1000 if n > 0 else 0,
            'correct': correct,
            'accuracy_pct': (correct / n * 100) if n > 0 else 0,
            'avg_candidates': avg_cand,
            'candidates_per_proc': avg_cand / size
        }

    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    base = Path(__file__).parent.parent.parent

    # Get size from command line argument
    if len(sys.argv) > 1:
        file_size = sys.argv[1]
    else:
        file_size = "250"

    if rank == 0:
        print("=" * 80)
        print(f"DICT-SPLIT MPI-{size} BENCHMARK - TURKISH DAMERAU-LEVENSHTEIN ({file_size})")
        print("=" * 80)

        print("\nLoading Turkish dictionary...")
        dict_tr = load_dictionary(str(base / "data/dictionary/tr_equal.txt"))
        print(f"  TR: {len(dict_tr):,} words")
        tr_by_len = group_by_length(dict_tr)

        gt_file = base / f"data/ground_truth/tr_hunspell_corrections_{file_size}.json"
        with open(gt_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        typos = list(ground_truth.keys())
        print(f"  Typos: {len(typos)}")
        sys.stdout.flush()
    else:
        tr_by_len = None
        ground_truth = {}
        typos = []

    tr_by_len = comm.bcast(tr_by_len, root=0)
    ground_truth = comm.bcast(ground_truth, root=0)
    typos = comm.bcast(typos, root=0)

    result = run_benchmark(comm, typos, tr_by_len, ground_truth)

    if rank == 0 and result:
        results_dir = base / "results" / "dict_split"
        results_dir.mkdir(parents=True, exist_ok=True)

        temp_file = results_dir / f"tr_damerau_dict_split_{file_size}_temp.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump({"num_procs": size, "results": [result]}, f, indent=2, ensure_ascii=False)
        print(f"\nTemp results saved to: {temp_file}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
