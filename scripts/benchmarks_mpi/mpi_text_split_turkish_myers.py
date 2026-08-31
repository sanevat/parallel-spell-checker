#!/usr/bin/env python3
"""
MPI Text-Split Spell Checker for Turkish - Myers Bit-Vector.
Runs one size at a time, saves to temp file for runner to collect.
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

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py not installed")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spell_checker.algorithms import myers_bitvector


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


def find_best_correction(typo: str, by_len: Dict[int, List[str]], dist_fn: Callable) -> Tuple[str, int]:
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
    results = []
    for typo in typos:
        best_word, num_cand = find_best_correction(typo, by_len, dist_fn)
        results.append((typo, best_word, num_cand))
    return results


def split_list(lst: List, n: int) -> List[List]:
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def run_text_split_benchmark(comm, typos, by_len, ground_truth):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"\n--- [TEXT-SPLIT MPI-{size}] Myers Bit-Vector + TR ({len(typos)} words) ---")
        sys.stdout.flush()
        chunks = split_list(typos, size)
        words_per_proc = len(typos) / size
    else:
        chunks = None
        words_per_proc = 0

    by_len = comm.bcast(by_len, root=0)
    local_typos = comm.scatter(chunks, root=0)

    comm.Barrier()
    start_time = time.perf_counter()
    local_results = process_chunk(local_typos, by_len, myers_bitvector)
    comm.Barrier()
    elapsed = time.perf_counter() - start_time

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        corrections = {}
        total_candidates = 0
        for proc_results in all_results:
            for typo, correction, num_cand in proc_results:
                corrections[typo] = correction
                total_candidates += num_cand

        correct = sum(1 for typo in typos if typo in ground_truth and corrections.get(typo) == ground_truth[typo])
        n = len(typos)
        ms_per_word = elapsed / n * 1000
        accuracy = correct / n * 100
        avg_cand = total_candidates / n

        print(f"  Completed: {n} words in {elapsed:.3f}s, {correct}/{n} correct ({accuracy:.1f}%), {ms_per_word:.2f}ms/word")
        sys.stdout.flush()

        return TextSplitResult(
            algorithm="Myers-BitVector", language="TR", num_procs=size, misspelled=n,
            total_time_s=elapsed, ms_per_word=ms_per_word, correct=correct,
            accuracy_pct=accuracy, avg_candidates=avg_cand, words_per_proc=words_per_proc
        )
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
        print(f"TEXT-SPLIT MPI-{size} BENCHMARK - TURKISH MYERS BIT-VECTOR ({file_size})")
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

    result = run_text_split_benchmark(comm, typos, tr_by_len, ground_truth)

    if rank == 0 and result:
        results_dir = base / "results" / "text_split"
        results_dir.mkdir(parents=True, exist_ok=True)

        temp_file = results_dir / f"tr_myers_text_split_{file_size}_temp.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump({"num_procs": size, "results": [asdict(result)]}, f, indent=2, ensure_ascii=False)
        print(f"\nTemp results saved to: {temp_file}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
