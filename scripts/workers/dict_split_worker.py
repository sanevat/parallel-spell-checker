#!/usr/bin/env python3
"""
MPI worker for Dict-Split Levenshtein on specified dataset size.
Strategy: Split CANDIDATES across processes for each typo word.
Usage: mpiexec -n N python dict_split_worker.py <dataset_size>
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
    candidates_per_proc: float


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
    best_dist = levenshtein(typo, candidates[0])

    for c in candidates[1:]:
        d = levenshtein(typo, c)
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


def run_benchmark(comm, typos, dict_by_len, ground_truth, lang):
    rank = comm.Get_rank()
    size = comm.Get_size()

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

    if rank == 0:
        n = len(typos)
        avg_cand = total_candidates / n if n > 0 else 0

        return Result(
            algorithm='Levenshtein',
            language=lang,
            num_procs=size,
            misspelled=n,
            total_time_s=total_time,
            ms_per_word=total_time / n * 1000 if n > 0 else 0,
            correct=correct,
            accuracy_pct=(correct / n * 100) if n > 0 else 0,
            avg_candidates=avg_cand,
            candidates_per_proc=avg_cand / size
        )

    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: mpiexec -n N python dict_split_worker.py <dataset_size>")
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

    mk_by_len = comm.bcast(mk_by_len, root=0)
    en_by_len = comm.bcast(en_by_len, root=0)
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
        with open(results_dir / "dict_split_temp.json", 'w', encoding='utf-8') as f:
            json.dump({"num_procs": size, "results": [asdict(r) for r in results]}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
