"""
MPI Text-Split Scalability Benchmark for Spell Checking

Measures performance of text-split parallelization approach
using MPI with different numbers of processes and word counts.
"""

import time
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict
import subprocess
import tempfile

# Check if running with MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    print("Warning: mpi4py not installed. Running in sequential mode only.")


def load_dictionary(lang: str = 'en') -> List[str]:
    """Load dictionary for specified language."""
    dict_path = Path(__file__).parent / 'data' / 'dictionary' / f'{lang}_equal.txt'
    if dict_path.exists():
        with open(dict_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def load_test_text(lang: str = 'en', size: str = '1MB') -> str:
    """Load test text for specified language and size."""
    lang_folder = 'english' if lang == 'en' else 'macedonian'
    text_path = Path(__file__).parent / 'data' / 'test_texts' / lang_folder / f'{lang}_hunspell_typos_{size}.txt'
    if text_path.exists():
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def split_text_into_words(text: str) -> List[str]:
    """Split text into words."""
    import re
    return re.findall(r'\b\w+\b', text.lower())


def check_word(word: str, dictionary: set, max_distance: int = 2) -> Tuple[str, bool, str]:
    """
    Check if word is in dictionary or find closest match.
    Returns: (word, is_correct, suggestion)
    """
    if word in dictionary:
        return (word, True, word)

    best_match = None
    best_dist = max_distance + 1

    for dict_word in dictionary:
        if abs(len(dict_word) - len(word)) > max_distance:
            continue
        dist = levenshtein_distance(word, dict_word)
        if dist < best_dist:
            best_dist = dist
            best_match = dict_word
        if dist == 0:
            break

    return (word, False, best_match if best_dist <= max_distance else None)


def process_words_sequential(words: List[str], dictionary: set) -> List[Tuple]:
    """Process words sequentially."""
    results = []
    for word in words:
        results.append(check_word(word, dictionary))
    return results


def process_words_mpi(words: List[str], dictionary: set):
    """Process words using MPI text-split approach."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split words among processes
    chunk_size = len(words) // size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < size - 1 else len(words)

    local_words = words[start_idx:end_idx]

    # Process local chunk
    local_results = []
    for word in local_words:
        local_results.append(check_word(word, dictionary))

    # Gather results
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Flatten results
        results = []
        for chunk in all_results:
            results.extend(chunk)
        return results
    return None


def benchmark_sequential(words: List[str], dictionary: set, num_runs: int = 3) -> float:
    """Benchmark sequential processing."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        process_words_sequential(words, dictionary)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return sum(times) / len(times)


def run_mpi_benchmark(script_path: str, num_processes: int, num_words: int, lang: str) -> float:
    """Run MPI benchmark with specified number of processes."""
    cmd = [
        'mpiexec', '-n', str(num_processes),
        sys.executable, script_path,
        '--mode', 'worker',
        '--num_words', str(num_words),
        '--lang', lang
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # Parse time from output
            for line in result.stdout.split('\n'):
                if line.startswith('TIME:'):
                    return float(line.split(':')[1].strip())
        else:
            print(f"Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"Timeout for {num_processes} processes, {num_words} words")
    except FileNotFoundError:
        print("mpiexec not found. Please install MPI.")

    return -1


def mpi_worker_mode(num_words: int, lang: str):
    """Run as MPI worker."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Load data on all processes
    dictionary = set(load_dictionary(lang))
    text = load_test_text(lang, '1MB')
    words = split_text_into_words(text)[:num_words]

    # Broadcast dictionary to all processes
    dictionary = comm.bcast(dictionary, root=0)
    words = comm.bcast(words, root=0)

    # Synchronize before timing
    comm.Barrier()

    start = time.perf_counter()
    results = process_words_mpi(words, dictionary)
    comm.Barrier()
    elapsed = time.perf_counter() - start

    if rank == 0:
        print(f"TIME:{elapsed}")


def run_full_benchmark():
    """Run full scalability benchmark and save results."""
    word_counts = [250, 1500, 2500, 3500, 4000, 4500, 5000]
    process_counts = [1, 2, 4, 8]
    languages = ['mk', 'en']

    results = {
        'word_counts': word_counts,
        'process_counts': process_counts,
        'data': {}
    }

    script_path = os.path.abspath(__file__)

    for lang in languages:
        lang_name = 'MK' if lang == 'mk' else 'EN'
        results['data'][lang_name] = {}

        # Load data for sequential benchmark
        dictionary = set(load_dictionary(lang))
        text = load_test_text(lang, '1MB')

        print(f"\nBenchmarking {lang_name}...")

        for num_words in word_counts:
            words = split_text_into_words(text)[:num_words]

            for num_procs in process_counts:
                key = f"{num_procs}_processes"
                if key not in results['data'][lang_name]:
                    results['data'][lang_name][key] = []

                if num_procs == 1:
                    # Sequential benchmark
                    elapsed = benchmark_sequential(words, dictionary)
                    print(f"  Sequential, {num_words} words: {elapsed:.2f}s")
                else:
                    # MPI benchmark
                    elapsed = run_mpi_benchmark(script_path, num_procs, num_words, lang)
                    print(f"  {num_procs} processes, {num_words} words: {elapsed:.2f}s")

                results['data'][lang_name][key].append(elapsed)

    # Save results
    results_path = Path(__file__).parent / 'benchmark_results_text_split_new.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MPI Text-Split Benchmark')
    parser.add_argument('--mode', choices=['benchmark', 'worker', 'plot'], default='benchmark')
    parser.add_argument('--num_words', type=int, default=1000)
    parser.add_argument('--lang', choices=['en', 'mk'], default='en')
    args = parser.parse_args()

    if args.mode == 'worker':
        if not HAS_MPI:
            print("ERROR: mpi4py required for worker mode")
            sys.exit(1)
        mpi_worker_mode(args.num_words, args.lang)
    elif args.mode == 'plot':
        from plot_text_split_scalability import plot_results
        plot_results()
    else:
        run_full_benchmark()


if __name__ == "__main__":
    main()
