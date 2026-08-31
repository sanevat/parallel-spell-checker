#!/usr/bin/env python3
"""
MPI worker for Myers Bit-Vector EN only at 5000 words.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    base = Path(__file__).parent.parent

    if rank == 0:
        print("=" * 80)
        print(f"Myers Bit-Vector EN - MPI-{size} (5000 words)")
        print("=" * 80)

        print("\nLoading EN dictionary...")
        dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
        print(f"  EN: {len(dict_en):,} words")

        en_by_len = group_by_length(dict_en)

        with open(base / "data/ground_truth/en_hunspell_corrections_5000.json", 'r', encoding='utf-8') as f:
            gt_en = json.load(f)

        typos_en = list(gt_en.keys())
        print(f"  EN typos: {len(typos_en)}")
        print(f"  Words per process: {len(typos_en) // size}")
        sys.stdout.flush()

        chunks = split_list(typos_en, size)
        words_per_proc = len(typos_en) / size
    else:
        en_by_len = gt_en = typos_en = None
        chunks = None
        words_per_proc = 0

    en_by_len = comm.bcast(en_by_len, root=0)
    gt_en = comm.bcast(gt_en, root=0)
    typos_en = comm.bcast(typos_en, root=0)
    local_typos = comm.scatter(chunks, root=0)

    if rank == 0:
        print(f"\nRunning Myers Bit-Vector...")
        sys.stdout.flush()

    comm.Barrier()
    start_time = time.perf_counter()
    local_results = process_chunk(local_typos, en_by_len, myers_bitvector)
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

        correct = sum(1 for typo in typos_en if typo in gt_en and corrections.get(typo) == gt_en[typo])
        n = len(typos_en)
        ms_per_word = elapsed / n * 1000
        accuracy = correct / n * 100
        avg_cand = total_candidates / n

        print(f"\nCompleted: {n} words in {elapsed:.1f}s")
        print(f"  {correct}/{n} correct ({accuracy:.1f}%)")
        print(f"  {ms_per_word:.1f}ms/word")

        result = TextSplitResult(
            algorithm="Myers Bit-Vector", language="EN", num_procs=size, misspelled=n,
            total_time_s=elapsed, ms_per_word=ms_per_word, correct=correct,
            accuracy_pct=accuracy, avg_candidates=avg_cand, words_per_proc=words_per_proc
        )

        results_dir = base / "results"
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "fix_myers_5000_temp.json", 'w', encoding='utf-8') as f:
            json.dump({"num_procs": size, "results": [asdict(result)]}, f, indent=2, ensure_ascii=False)

        print(f"\nTemp results saved.")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
