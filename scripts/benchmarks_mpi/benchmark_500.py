#!/usr/bin/env python3
"""
Generate 500-word test files and run full benchmark (Sequential + MPI).
"""

import os
import sys
import json
import time
import random
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector

BASE = Path(__file__).parent.parent

# Error type distributions (same as 2K)
ERROR_TYPES = {
    'insert': 0.351,
    'replace': 0.308,
    'swap': 0.179,
    'delete': 0.162
}

MK_CHARS = 'абвгдѓежзѕијклљмнњопрстќуфхцчџш'
EN_CHARS = 'abcdefghijklmnopqrstuvwxyz'


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


def generate_typo(word: str, chars: str) -> Tuple[str, str]:
    """Generate a typo and return (typo, error_type)."""
    if len(word) < 3:
        return word, 'none'

    r = random.random()
    cumulative = 0
    error_type = 'insert'

    for etype, prob in ERROR_TYPES.items():
        cumulative += prob
        if r < cumulative:
            error_type = etype
            break

    word_list = list(word)

    if error_type == 'insert':
        pos = random.randint(0, len(word))
        char = random.choice(chars)
        word_list.insert(pos, char)
    elif error_type == 'delete' and len(word) > 3:
        pos = random.randint(0, len(word) - 1)
        del word_list[pos]
    elif error_type == 'replace':
        pos = random.randint(0, len(word) - 1)
        char = random.choice(chars)
        word_list[pos] = char
    elif error_type == 'swap' and len(word) > 1:
        pos = random.randint(0, len(word) - 2)
        word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]

    return ''.join(word_list), error_type


def generate_test_files(num_typos: int = 500):
    """Generate test files with specified number of typos."""
    print("=" * 80)
    print(f"GENERATING {num_typos}-WORD TEST FILES")
    print("=" * 80)

    # Load dictionaries
    dict_mk = load_dictionary(str(BASE / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(BASE / "data/dictionary/en_equal.txt"))
    print(f"Loaded MK: {len(dict_mk):,} words, EN: {len(dict_en):,} words")

    dict_mk_set = set(dict_mk)
    dict_en_set = set(dict_en)

    # Pre-group by length
    mk_by_len = group_by_length(dict_mk)
    en_by_len = group_by_length(dict_en)

    # Filter words that have candidates in dictionary (length 4-12)
    mk_valid = [w for w in dict_mk if 4 <= len(w) <= 12]
    en_valid = [w for w in dict_en if 4 <= len(w) <= 12]

    random.shuffle(mk_valid)
    random.shuffle(en_valid)

    results = {}

    for lang, valid_words, dict_set, by_len, chars in [
        ('mk', mk_valid, dict_mk_set, mk_by_len, MK_CHARS),
        ('en', en_valid, dict_en_set, en_by_len, EN_CHARS)
    ]:
        print(f"\nGenerating {lang.upper()} test file...")

        typos = []
        corrections = {}
        error_counts = defaultdict(int)
        attempts = 0
        max_attempts = num_typos * 50

        for word in valid_words:
            if len(typos) >= num_typos:
                break
            if attempts > max_attempts:
                break
            attempts += 1

            typo, error_type = generate_typo(word, chars)

            # Verify: typo NOT in dict, original IS in dict
            if typo not in dict_set and word in dict_set and typo != word:
                # Verify correction is in candidates
                candidates = get_candidates(len(typo), by_len)
                if word in candidates:
                    typos.append(typo)
                    corrections[typo] = word
                    error_counts[error_type] += 1

        # Add correct words (~1.857 ratio like 2K: 3714/2000)
        num_correct = int(num_typos * 1.857)
        correct_words = [w for w in valid_words if w in dict_set and w not in corrections.values()][:num_correct]

        # Build text
        all_words = typos + correct_words
        random.shuffle(all_words)

        # Create sentences
        text_lines = []
        words_per_line = 10
        for i in range(0, len(all_words), words_per_line):
            line = ' '.join(all_words[i:i+words_per_line])
            text_lines.append(line + '.')

        text = '\n'.join(text_lines)

        # Save files
        typo_file = BASE / f"data/test_texts/{'macedonian' if lang == 'mk' else 'english'}/{lang}_hunspell_typos_500.txt"
        gt_file = BASE / f"data/ground_truth/{lang}_hunspell_corrections_500.json"

        with open(typo_file, 'w', encoding='utf-8') as f:
            f.write(text)

        with open(gt_file, 'w', encoding='utf-8') as f:
            json.dump(corrections, f, ensure_ascii=False, indent=2)

        print(f"  Typos: {len(typos)}, Correct: {len(correct_words)}, Total: {len(all_words)}")
        print(f"  Error types: {dict(error_counts)}")
        print(f"  Saved: {typo_file.name}, {gt_file.name}")

        results[lang] = {
            'typos': len(typos),
            'correct': len(correct_words),
            'total': len(all_words),
            'error_types': dict(error_counts),
            'corrections': corrections
        }

    return results


def run_sequential_benchmark(typos_mk, typos_en, mk_by_len, en_by_len, gt_mk, gt_en):
    """Run sequential benchmark."""
    print("\n" + "=" * 80)
    print("SEQUENTIAL BENCHMARK")
    print("=" * 80)

    algorithms = [
        ("Levenshtein", levenshtein),
        ("Damerau-Levenshtein", damerau_levenshtein),
        ("Myers Bit-Vector", myers_bitvector),
    ]

    results = []

    for algo_name, algo_fn in algorithms:
        for lang, typos, by_len, gt in [
            ("MK", typos_mk, mk_by_len, gt_mk),
            ("EN", typos_en, en_by_len, gt_en)
        ]:
            print(f"\n--- Sequential {algo_name} + {lang} ({len(typos)} words) ---")
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
                    d = algo_fn(typo, c)
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

            print(f"  Completed: {len(typos)} words in {total_time:.1f}s, "
                  f"{correct}/{len(typos)} correct ({accuracy:.1f}%)")
            sys.stdout.flush()

            results.append({
                'algorithm': algo_name,
                'language': lang,
                'num_procs': 1,
                'misspelled': len(typos),
                'total_time_s': total_time,
                'ms_per_word': ms_per_word,
                'correct': correct,
                'accuracy_pct': accuracy,
                'avg_candidates': total_candidates / len(typos)
            })

    return results


def run_mpi_benchmark(num_procs: int):
    """Run MPI benchmark with specified number of processes."""
    print(f"\n" + "=" * 80)
    print(f"MPI-{num_procs} BENCHMARK")
    print("=" * 80)
    sys.stdout.flush()

    # Create a temporary MPI script for 500-word files
    mpi_script = BASE / "spell_checker" / "mpi_benchmark_500.py"

    cmd = ["mpiexec", "-n", str(num_procs), "python", "-u", str(mpi_script)]

    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(BASE))

    # Read results from file
    results_file = BASE / "results" / "mpi_500_temp.json"
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('results', [])

    return []


def main():
    random.seed(42)

    # Step 1: Generate test files
    gen_results = generate_test_files(500)

    # Load data for sequential benchmark
    print("\n" + "=" * 80)
    print("LOADING DATA FOR BENCHMARKS")
    print("=" * 80)

    dict_mk = load_dictionary(str(BASE / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(BASE / "data/dictionary/en_equal.txt"))

    mk_by_len = group_by_length(dict_mk)
    en_by_len = group_by_length(dict_en)

    # Load ground truth
    with open(BASE / "data/ground_truth/mk_hunspell_corrections_500.json", 'r', encoding='utf-8') as f:
        gt_mk = json.load(f)
    with open(BASE / "data/ground_truth/en_hunspell_corrections_500.json", 'r', encoding='utf-8') as f:
        gt_en = json.load(f)

    # Extract typos
    typos_mk = list(gt_mk.keys())
    typos_en = list(gt_en.keys())

    print(f"MK typos: {len(typos_mk)}, EN typos: {len(typos_en)}")

    all_results = {
        'description': 'Scalability benchmark with 500 typos, 50K dict',
        'generation': {
            'mk': {k: v for k, v in gen_results['mk'].items() if k != 'corrections'},
            'en': {k: v for k, v in gen_results['en'].items() if k != 'corrections'}
        },
        'sequential': [],
        'mpi': {}
    }

    # Step 2: Run sequential benchmark
    seq_results = run_sequential_benchmark(typos_mk, typos_en, mk_by_len, en_by_len, gt_mk, gt_en)
    all_results['sequential'] = seq_results

    # Build baseline for speedup calculation
    seq_baseline = {}
    for r in seq_results:
        seq_baseline[(r['algorithm'], r['language'])] = r['ms_per_word']

    # Save sequential results immediately
    results_file = BASE / "results" / "scalability_500.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSequential results saved to: {results_file}")

    # Step 3: Run MPI benchmarks_mpi
    for num_procs in [2, 4, 8]:
        print(f"\n{'='*80}")
        print(f"RUNNING MPI-{num_procs} BENCHMARK")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Run MPI benchmark via subprocess
        mpi_script = BASE / "spell_checker" / "mpi_benchmark_500.py"
        cmd = ["mpiexec", "-n", str(num_procs), "python", "-u", str(mpi_script)]

        subprocess.run(cmd, cwd=str(BASE))

        # Read results
        temp_file = BASE / "results" / "mpi_500_temp.json"
        if temp_file.exists():
            with open(temp_file, 'r', encoding='utf-8') as f:
                mpi_data = json.load(f)
            all_results['mpi'][str(num_procs)] = mpi_data.get('results', [])

    # Save final results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print final comparison table
    print("\n" + "=" * 100)
    print("FINAL COMPARISON TABLE (500 words)")
    print("=" * 100)
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Seq(ms)':<8} | {'MPI-2':<8} | {'MPI-4':<8} | {'MPI-8':<8} | {'Speedup-8':<10}")
    print("-" * 100)

    for algo in ["Levenshtein", "Damerau-Levenshtein", "Myers Bit-Vector"]:
        for lang in ["MK", "EN"]:
            seq_ms = seq_baseline.get((algo, lang), 0)

            mpi2_ms = 0
            mpi4_ms = 0
            mpi8_ms = 0

            for r in all_results['mpi'].get('2', []):
                if r['algorithm'] == algo and r['language'] == lang:
                    mpi2_ms = r['ms_per_word']
            for r in all_results['mpi'].get('4', []):
                if r['algorithm'] == algo and r['language'] == lang:
                    mpi4_ms = r['ms_per_word']
            for r in all_results['mpi'].get('8', []):
                if r['algorithm'] == algo and r['language'] == lang:
                    mpi8_ms = r['ms_per_word']

            speedup = seq_ms / mpi8_ms if mpi8_ms > 0 else 0

            print(f"{algo:<22} | {lang:<4} | {seq_ms:>8.1f} | {mpi2_ms:>8.1f} | {mpi4_ms:>8.1f} | {mpi8_ms:>8.1f} | {speedup:>8.2f}x")

    print("-" * 100)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
