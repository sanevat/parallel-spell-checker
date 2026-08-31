#!/usr/bin/env python3
"""
MPI worker for Levenshtein on specified dataset size.
Usage: mpiexec -n N python levenshtein_worker.py <dataset_size>
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
from spell_checker.algorithms import levenshtein


@dataclass
class Result:
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


def run_benchmark(comm, typos, by_len, ground_truth, lang):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        chunks = split_list(typos, size)
        words_per_proc = len(typos) / size
    else:
        chunks = None
        words_per_proc = 0

    by_len = comm.bcast(by_len, root=0)
    local_typos = comm.scatter(chunks, root=0)

    comm.Barrier()
    start_time = time.perf_counter()
    local_results = process_chunk(local_typos, by_len, levenshtein)
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

        return Result(
            algorithm="Levenshtein", language=lang, num_procs=size, misspelled=n,
            total_time_s=elapsed, ms_per_word=ms_per_word, correct=correct,
            accuracy_pct=accuracy, avg_candidates=avg_cand, words_per_proc=words_per_proc
        )
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: mpiexec -n N python levenshtein_worker.py <dataset_size>")
        sys.exit(1)

    dataset_size = sys.argv[1]

    # Map dataset size to file suffix
    size_map = {
        '250': '250', '500': '500', '1K': '1K', '1000': '1K',
        '1500': '1500', '2K': '2K', '2000': '2K',
        '2500': '2500', '3500': '3500', '5000': '5000'
    }
    file_suffix = size_map.get(dataset_size, dataset_size)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    base = Path(__file__).parent.parent

    if rank == 0:
        dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
        dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))

        mk_by_len = group_by_length(dict_mk)
        en_by_len = group_by_length(dict_en)

        gt_mk_file = base / f"data/ground_truth/mk_hunspell_corrections_{file_suffix}.json"
        gt_en_file = base / f"data/ground_truth/en_hunspell_corrections_{file_suffix}.json"

        with open(gt_mk_file, 'r', encoding='utf-8') as f:
            gt_mk = json.load(f)
        with open(gt_en_file, 'r', encoding='utf-8') as f:
            gt_en = json.load(f)

        typos_mk = list(gt_mk.keys())
        typos_en = list(gt_en.keys())
    else:
        mk_by_len = en_by_len = gt_mk = gt_en = typos_mk = typos_en = None

    gt_mk = comm.bcast(gt_mk, root=0)
    gt_en = comm.bcast(gt_en, root=0)
    typos_mk = comm.bcast(typos_mk, root=0)
    typos_en = comm.bcast(typos_en, root=0)

    results = []

    r = run_benchmark(comm, typos_mk, mk_by_len, gt_mk, "MK")
    if r:
        results.append(r)

    r = run_benchmark(comm, typos_en, en_by_len, gt_en, "EN")
    if r:
        results.append(r)

    if rank == 0:
        results_dir = base / "results"
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "levenshtein_temp.json", 'w', encoding='utf-8') as f:
            json.dump({"num_procs": size, "results": [asdict(r) for r in results]}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
