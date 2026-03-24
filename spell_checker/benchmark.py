"""
Benchmark suite for comparing edit distance algorithms.

Measures and compares performance of Levenshtein, Damerau-Levenshtein,
and Myers' bit-vector algorithms on both English and Macedonian text.
"""

import time
import random
import string
from typing import List, Dict, Callable, Tuple
from dataclasses import dataclass
import statistics

from algorithms import levenshtein, damerau_levenshtein, myers_bitvector


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    algorithm: str
    language: str
    num_pairs: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    ops_per_second: float


def generate_english_words(count: int, min_len: int = 3, max_len: int = 15) -> List[str]:
    """Generate random English words."""
    words = []
    for _ in range(count):
        length = random.randint(min_len, max_len)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    return words


def generate_macedonian_words(count: int, min_len: int = 3, max_len: int = 15) -> List[str]:
    """Generate random Macedonian (Cyrillic) words."""
    # Macedonian Cyrillic alphabet (lowercase)
    macedonian_letters = 'абвгдѓежзѕијклљмнњопрстќуфхцчџш'

    words = []
    for _ in range(count):
        length = random.randint(min_len, max_len)
        word = ''.join(random.choices(macedonian_letters, k=length))
        words.append(word)
    return words


def generate_word_pairs(words: List[str], count: int) -> List[Tuple[str, str]]:
    """Generate pairs of words for comparison."""
    pairs = []
    for _ in range(count):
        w1, w2 = random.sample(words, 2)
        pairs.append((w1, w2))
    return pairs


def benchmark_algorithm(
    algorithm: Callable[[str, str], int],
    pairs: List[Tuple[str, str]],
    algorithm_name: str,
    language: str,
    warmup_runs: int = 10
) -> BenchmarkResult:
    """
    Benchmark a single algorithm on a set of word pairs.

    Args:
        algorithm: Edit distance function
        pairs: List of (word1, word2) tuples
        algorithm_name: Name for reporting
        language: Language label
        warmup_runs: Number of warmup iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for w1, w2 in pairs[:warmup_runs]:
        algorithm(w1, w2)

    # Actual benchmark
    times = []

    for w1, w2 in pairs:
        start = time.perf_counter()
        algorithm(w1, w2)
        end = time.perf_counter()
        times.append(end - start)

    total_time = sum(times)
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    ops_per_second = len(pairs) / total_time if total_time > 0 else float('inf')

    return BenchmarkResult(
        algorithm=algorithm_name,
        language=language,
        num_pairs=len(pairs),
        total_time=total_time,
        avg_time=avg_time,
        min_time=min_time,
        max_time=max_time,
        std_dev=std_dev,
        ops_per_second=ops_per_second
    )


def run_benchmarks(
    num_words: int = 100,
    num_pairs: int = 1000,
    word_lengths: Tuple[int, int] = (5, 15)
) -> Dict[str, List[BenchmarkResult]]:
    """
    Run comprehensive benchmarks_mpi for all algorithms.

    Args:
        num_words: Number of words to generate
        num_pairs: Number of word pairs to compare
        word_lengths: (min_length, max_length) tuple

    Returns:
        Dictionary mapping language to list of results
    """
    min_len, max_len = word_lengths

    print(f"Generating test data...")
    print(f"  - {num_words} words per language")
    print(f"  - {num_pairs} comparison pairs")
    print(f"  - Word lengths: {min_len}-{max_len} characters")
    print()

    # Generate test data
    english_words = generate_english_words(num_words, min_len, max_len)
    macedonian_words = generate_macedonian_words(num_words, min_len, max_len)

    english_pairs = generate_word_pairs(english_words, num_pairs)
    macedonian_pairs = generate_word_pairs(macedonian_words, num_pairs)

    algorithms = [
        (levenshtein, "Levenshtein"),
        (damerau_levenshtein, "Damerau-Levenshtein"),
        (myers_bitvector, "Myers Bit-Vector"),
    ]

    results = {"English": [], "Macedonian": []}

    # Benchmark English
    print("Benchmarking English (ASCII)...")
    for algo_fn, algo_name in algorithms:
        result = benchmark_algorithm(algo_fn, english_pairs, algo_name, "English")
        results["English"].append(result)
        print(f"  {algo_name}: {result.ops_per_second:.0f} ops/sec")

    print()

    # Benchmark Macedonian
    print("Benchmarking Macedonian (Cyrillic UTF-8)...")
    for algo_fn, algo_name in algorithms:
        result = benchmark_algorithm(algo_fn, macedonian_pairs, algo_name, "Macedonian")
        results["Macedonian"].append(result)
        print(f"  {algo_name}: {result.ops_per_second:.0f} ops/sec")

    return results


def print_detailed_results(results: Dict[str, List[BenchmarkResult]]) -> None:
    """Print detailed benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("DETAILED BENCHMARK RESULTS")
    print("=" * 80)

    for language, lang_results in results.items():
        print(f"\n{language}:")
        print("-" * 75)
        print(f"{'Algorithm':<22} {'Ops/sec':>10} {'Avg (us)':>10} {'Min (us)':>10} "
              f"{'Max (us)':>10} {'StdDev':>10}")
        print("-" * 75)

        for r in lang_results:
            print(f"{r.algorithm:<22} {r.ops_per_second:>10.0f} "
                  f"{r.avg_time*1e6:>10.2f} {r.min_time*1e6:>10.2f} "
                  f"{r.max_time*1e6:>10.2f} {r.std_dev*1e6:>10.2f}")


def print_comparison_summary(results: Dict[str, List[BenchmarkResult]]) -> None:
    """Print a summary comparison of algorithms and languages."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Compare UTF-8 overhead
    print("\nUTF-8 Overhead (Macedonian vs English):")
    print("-" * 50)

    eng_results = {r.algorithm: r for r in results["English"]}
    mk_results = {r.algorithm: r for r in results["Macedonian"]}

    for algo in eng_results:
        eng_ops = eng_results[algo].ops_per_second
        mk_ops = mk_results[algo].ops_per_second
        overhead = ((eng_ops - mk_ops) / eng_ops) * 100 if eng_ops > 0 else 0

        print(f"  {algo:<22}: {overhead:+.1f}% slower with Cyrillic")

    # Find fastest algorithm per language
    print("\nFastest Algorithm:")
    print("-" * 50)

    for language, lang_results in results.items():
        fastest = max(lang_results, key=lambda r: r.ops_per_second)
        print(f"  {language}: {fastest.algorithm} ({fastest.ops_per_second:.0f} ops/sec)")


def benchmark_by_length(
    lengths: List[int] = [5, 10, 20, 50, 100],
    pairs_per_length: int = 500
) -> None:
    """
    Benchmark algorithms across different string lengths.

    Shows how performance scales with string length.
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE BY STRING LENGTH")
    print("=" * 80)

    algorithms = [
        (levenshtein, "Levenshtein"),
        (damerau_levenshtein, "Damerau-Lev"),
        (myers_bitvector, "Myers"),
    ]

    print(f"\n{'Length':<8}", end="")
    for _, name in algorithms:
        print(f"{name:>15}", end="")
    print()
    print("-" * 53)

    for length in lengths:
        print(f"{length:<8}", end="")

        # Generate words of specific length
        words = generate_english_words(100, length, length)
        pairs = generate_word_pairs(words, pairs_per_length)

        for algo_fn, _ in algorithms:
            start = time.perf_counter()
            for w1, w2 in pairs:
                algo_fn(w1, w2)
            elapsed = time.perf_counter() - start
            ops_per_sec = pairs_per_length / elapsed
            print(f"{ops_per_sec:>12.0f} op/s", end="")

        print()


def run_spell_check_benchmark(
    dictionary_sizes: List[int] = [100, 1000, 10000],
    num_queries: int = 100
) -> None:
    """
    Benchmark spell checking performance with different dictionary sizes.
    """
    print("\n" + "=" * 80)
    print("SPELL CHECK BENCHMARK (Dictionary Lookup)")
    print("=" * 80)

    algorithms = [
        (levenshtein, "Levenshtein"),
        (damerau_levenshtein, "Damerau-Lev"),
        (myers_bitvector, "Myers"),
    ]

    for dict_size in dictionary_sizes:
        print(f"\nDictionary size: {dict_size} words")
        print("-" * 50)

        dictionary = generate_english_words(dict_size, 5, 12)
        queries = generate_english_words(num_queries, 5, 12)

        for algo_fn, algo_name in algorithms:
            start = time.perf_counter()

            for query in queries:
                best_match = None
                best_dist = float('inf')
                for word in dictionary:
                    dist = algo_fn(query, word)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = word
                        if dist == 0:
                            break

            elapsed = time.perf_counter() - start
            queries_per_sec = num_queries / elapsed

            print(f"  {algo_name:<20}: {queries_per_sec:.1f} queries/sec")


def main():
    """Run all benchmarks_mpi."""
    print("=" * 80)
    print("EDIT DISTANCE ALGORITHM BENCHMARKS")
    print("Sequential Implementation - Python")
    print("=" * 80)

    # Set random seed for reproducibility
    random.seed(42)

    # Main benchmark
    results = run_benchmarks(
        num_words=200,
        num_pairs=2000,
        word_lengths=(5, 15)
    )

    print_detailed_results(results)
    print_comparison_summary(results)

    # Performance by length
    benchmark_by_length()

    # Spell check simulation
    run_spell_check_benchmark(
        dictionary_sizes=[100, 500, 1000],
        num_queries=50
    )

    print("\n" + "=" * 80)
    print("Benchmarks complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
