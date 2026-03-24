#!/usr/bin/env python3
"""
Run Levenshtein dict_split benchmark for 1, 2, 4, 8 processes.
Runs each configuration 2 times and keeps the smaller (better) result.
Saves results to results/dict_split_levenshtein.json in text_split format.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spell_checker.algorithms import levenshtein

BASE = Path(__file__).parent.parent


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


def run_sequential_levenshtein(typos, by_len, gt, lang):
    """Run sequential (1 process) Levenshtein benchmark."""
    print(f"\n--- Sequential Levenshtein + {lang} ({len(typos)} words) ---")
    sys.stdout.flush()

    correct = 0
    total_time = 0
    total_candidates = 0

    for i, typo in enumerate(typos):
        candidates = get_candidates(len(typo), by_len)
        total_candidates += len(candidates)

        start = time.perf_counter()
        best_word = ""
        best_dist = 999999
        for c in candidates:
            d = levenshtein(typo, c)
            if d < best_dist:
                best_dist = d
                best_word = c
                if d == 0:
                    break
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if typo in gt and best_word == gt[typo]:
            correct += 1

        if (i + 1) % 50 == 0:
            avg_ms = total_time / (i + 1) * 1000
            print(f"  Progress: {i+1}/{len(typos)}, elapsed: {total_time:.1f}s, avg: {avg_ms:.1f}ms/word")
            sys.stdout.flush()

    ms_per_word = total_time / len(typos) * 1000
    accuracy = correct / len(typos) * 100
    avg_cand = total_candidates / len(typos)

    print(f"  Completed: {len(typos)} words in {total_time:.1f}s, "
          f"{correct}/{len(typos)} correct ({accuracy:.1f}%)")
    sys.stdout.flush()

    return {
        'algorithm': 'Levenshtein',
        'language': lang,
        'num_procs': 1,
        'misspelled': len(typos),
        'total_time_s': total_time,
        'ms_per_word': ms_per_word,
        'correct': correct,
        'accuracy_pct': accuracy,
        'avg_candidates': avg_cand,
        'candidates_per_proc': avg_cand
    }


def run_mpi_levenshtein(num_procs):
    """Run MPI Levenshtein benchmark via subprocess."""
    print(f"\n--- Running MPI-{num_procs} ---")
    sys.stdout.flush()

    mpi_script = BASE / "spell_checker" / "mpi_dict_split_levenshtein.py"
    cmd = ["mpiexec", "-n", str(num_procs), "python", "-u", str(mpi_script)]

    subprocess.run(cmd, cwd=str(BASE))

    temp_file = BASE / "results" / "dict_split_levenshtein_temp.json"
    if temp_file.exists():
        with open(temp_file, 'r', encoding='utf-8') as f:
            return json.load(f).get('results', [])
    return []


def main():
    print("=" * 80)
    print("LEVENSHTEIN DICT-SPLIT BENCHMARK (2 runs, keep best)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    dict_mk = load_dictionary(str(BASE / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(BASE / "data/dictionary/en_equal.txt"))

    mk_by_len = group_by_length(dict_mk)
    en_by_len = group_by_length(dict_en)

    with open(BASE / "data/ground_truth/mk_hunspell_corrections_250.json", 'r', encoding='utf-8') as f:
        gt_mk = json.load(f)
    with open(BASE / "data/ground_truth/en_hunspell_corrections_250.json", 'r', encoding='utf-8') as f:
        gt_en = json.load(f)

    typos_mk = list(gt_mk.keys())
    typos_en = list(gt_en.keys())

    print(f"MK typos: {len(typos_mk)}, EN typos: {len(typos_en)}")
    print(f"MK dict: {len(dict_mk):,} words, EN dict: {len(dict_en):,} words")

    all_results = {
        'description': 'DICT-SPLIT benchmark for Levenshtein (250 words, 2 runs, best kept)',
        'dict_split': {}
    }

    for num_procs in [1, 2, 4, 8]:
        print(f"\n{'='*80}")
        print(f"PROCESS COUNT: {num_procs}")
        print(f"{'='*80}")

        best_results = {}

        for run_num in [1, 2]:
            print(f"\n>>> RUN {run_num}/2 <<<")

            if num_procs == 1:
                results = []
                for lang, typos, by_len, gt in [
                    ("MK", typos_mk, mk_by_len, gt_mk),
                    ("EN", typos_en, en_by_len, gt_en)
                ]:
                    r = run_sequential_levenshtein(typos, by_len, gt, lang)
                    results.append(r)
            else:
                results = run_mpi_levenshtein(num_procs)

            for r in results:
                key = r['language']
                if key not in best_results:
                    best_results[key] = r
                    print(f"  {key}: First run = {r['total_time_s']:.2f}s")
                else:
                    if r['total_time_s'] < best_results[key]['total_time_s']:
                        print(f"  {key}: Run {run_num} BETTER: {r['total_time_s']:.2f}s < {best_results[key]['total_time_s']:.2f}s")
                        best_results[key] = r
                    else:
                        print(f"  {key}: Run 1 still best: {best_results[key]['total_time_s']:.2f}s <= {r['total_time_s']:.2f}s")

        all_results['dict_split'][str(num_procs)] = list(best_results.values())

    # Save results
    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "dict_split_levenshtein.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 100)
    print("FINAL RESULTS - LEVENSHTEIN DICT-SPLIT (Best of 2 runs)")
    print("=" * 100)
    print(f"{'Lang':<6} | {'1 Proc':>12} | {'2 Proc':>12} | {'4 Proc':>12} | {'8 Proc':>12} | {'Speedup 8x':>12}")
    print("-" * 100)

    for lang in ["MK", "EN"]:
        row = f"{lang:<6} |"
        time_1 = 0
        for np in ['1', '2', '4', '8']:
            for r in all_results['dict_split'].get(np, []):
                if r['language'] == lang:
                    row += f" {r['total_time_s']:>10.2f}s |"
                    if np == '1':
                        time_1 = r['total_time_s']
                    if np == '8':
                        speedup = time_1 / r['total_time_s'] if r['total_time_s'] > 0 else 0
                        row += f" {speedup:>10.2f}x"
        print(row)

    print("-" * 100)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
