#!/usr/bin/env python3
"""
C-like baseline using Numba JIT compilation.
Compiles Python to native machine code via LLVM for fair CUDA comparison.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from numba import jit, prange
from numba.typed import List as NumbaList

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


@jit(nopython=True, cache=True)
def levenshtein_jit(s1: np.ndarray, len1: int, s2: np.ndarray, len2: int) -> int:
    """Levenshtein distance - JIT compiled to native code."""
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Use 1D array for DP (previous row)
    prev = np.zeros(len2 + 1, dtype=np.int32)
    curr = np.zeros(len2 + 1, dtype=np.int32)

    for j in range(len2 + 1):
        prev[j] = j

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    return prev[len2]


@jit(nopython=True, cache=True)
def damerau_jit(s1: np.ndarray, len1: int, s2: np.ndarray, len2: int) -> int:
    """Damerau-Levenshtein distance - JIT compiled."""
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    prev2 = np.zeros(len2 + 1, dtype=np.int32)
    prev = np.zeros(len2 + 1, dtype=np.int32)
    curr = np.zeros(len2 + 1, dtype=np.int32)

    for j in range(len2 + 1):
        prev[j] = j

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            # Standard operations
            min_cost = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)

            # Transposition
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                min_cost = min(min_cost, prev2[j - 2] + cost)

            curr[j] = min_cost

        # Rotate arrays
        prev2, prev, curr = prev, curr, prev2

    return prev[len2]


@jit(nopython=True, cache=True)
def myers_jit(s1: np.ndarray, len1: int, s2: np.ndarray, len2: int) -> int:
    """Myers bit-vector algorithm - JIT compiled."""
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # For simplicity, use standard DP for words > 64 chars
    if len1 > 64:
        return levenshtein_jit(s1, len1, s2, len2)

    # Build Peq for pattern (s1) - use smaller array for common chars only
    # Map characters to indices 0-255 (hash by masking)
    Peq = np.zeros(256, dtype=np.uint64)
    for i in range(len1):
        c = s1[i] & 0xFF
        Peq[c] |= np.uint64(1) << np.uint64(i)

    # Myers algorithm
    Pv = np.uint64(0xFFFFFFFFFFFFFFFF)
    Mv = np.uint64(0)
    score = len1
    last_bit = np.uint64(1) << np.uint64(len1 - 1)

    for j in range(len2):
        c = s2[j] & 0xFF
        Eq = Peq[c]

        Xv = Eq | Mv
        Xh = ((Eq & Pv) + Pv) ^ Pv | Eq

        Ph = Mv | ~(Xh | Pv)
        Mh = Pv & Xh

        if Ph & last_bit:
            score += 1
        if Mh & last_bit:
            score -= 1

        Ph = (Ph << np.uint64(1)) | np.uint64(1)
        Mh = Mh << np.uint64(1)

        Pv = Mh | ~(Xv | Ph)
        Mv = Ph & Xv

    return score


@jit(nopython=True, cache=True)
def find_best_match_lev(typo: np.ndarray, typo_len: int,
                        dict_chars: np.ndarray, dict_offsets: np.ndarray,
                        dict_lengths: np.ndarray, cand_indices: np.ndarray,
                        num_candidates: int) -> Tuple[int, int]:
    """Find best match using Levenshtein - JIT compiled.
    Uses original dictionary arrays directly, only indices change per word.
    """
    best_idx = 0
    best_dist = 999999

    for i in range(num_candidates):
        idx = cand_indices[i]
        start = dict_offsets[idx]
        cand_len = dict_lengths[idx]
        cand = dict_chars[start:start + cand_len]

        dist = levenshtein_jit(typo, typo_len, cand, cand_len)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            if dist == 0:
                break

    return best_idx, best_dist


@jit(nopython=True, cache=True)
def find_best_match_dam(typo: np.ndarray, typo_len: int,
                        dict_chars: np.ndarray, dict_offsets: np.ndarray,
                        dict_lengths: np.ndarray, cand_indices: np.ndarray,
                        num_candidates: int) -> Tuple[int, int]:
    """Find best match using Damerau-Levenshtein - JIT compiled."""
    best_idx = 0
    best_dist = 999999

    for i in range(num_candidates):
        idx = cand_indices[i]
        start = dict_offsets[idx]
        cand_len = dict_lengths[idx]
        cand = dict_chars[start:start + cand_len]

        dist = damerau_jit(typo, typo_len, cand, cand_len)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            if dist == 0:
                break

    return best_idx, best_dist


@jit(nopython=True, cache=True)
def find_best_match_myers(typo: np.ndarray, typo_len: int,
                          dict_chars: np.ndarray, dict_offsets: np.ndarray,
                          dict_lengths: np.ndarray, cand_indices: np.ndarray,
                          num_candidates: int) -> Tuple[int, int]:
    """Find best match using Myers - JIT compiled."""
    best_idx = 0
    best_dist = 999999

    for i in range(num_candidates):
        idx = cand_indices[i]
        start = dict_offsets[idx]
        cand_len = dict_lengths[idx]
        cand = dict_chars[start:start + cand_len]

        dist = myers_jit(typo, typo_len, cand, cand_len)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            if dist == 0:
                break

    return best_idx, best_dist


class CBaselineChecker:
    """C-like baseline spell checker using Numba JIT."""

    def __init__(self, dictionary: List[str]):
        """Pre-encode dictionary once (like CUDA pre-transfer)."""
        all_chars = []
        offsets = []
        lengths = []
        offset = 0

        for word in dictionary:
            encoded = [ord(c) for c in word]
            all_chars.extend(encoded)
            offsets.append(offset)
            lengths.append(len(word))
            offset += len(word)

        self.dictionary = dictionary
        self.dict_chars = np.array(all_chars, dtype=np.int32)
        self.dict_offsets = np.array(offsets, dtype=np.int32)
        self.dict_lengths = np.array(lengths, dtype=np.int32)

        # Group by length for candidate filtering
        self.by_length = defaultdict(list)
        for i, word in enumerate(dictionary):
            self.by_length[len(word)].append(i)

    def _get_candidate_indices(self, typo_len: int, tol: int = 2) -> np.ndarray:
        """Get candidate indices within length tolerance (no data copying)."""
        indices = []
        for length in range(max(1, typo_len - tol), typo_len + tol + 1):
            if length in self.by_length:
                indices.extend(self.by_length[length])

        if not indices:
            return None

        return np.array(indices, dtype=np.int32)

    def find_correction(self, typo: str, algorithm: str = 'levenshtein') -> Tuple[str, int]:
        """Find best correction."""
        typo_encoded = np.array([ord(c) for c in typo], dtype=np.int32)
        typo_len = len(typo)

        # Get just the indices (no data copying!)
        cand_indices = self._get_candidate_indices(typo_len)
        if cand_indices is None:
            return ("", 999999)

        num_cands = len(cand_indices)

        # Pass original dictionary arrays + indices to JIT functions
        if algorithm == 'levenshtein':
            best_idx, best_dist = find_best_match_lev(
                typo_encoded, typo_len,
                self.dict_chars, self.dict_offsets, self.dict_lengths,
                cand_indices, num_cands)
        elif algorithm == 'damerau':
            best_idx, best_dist = find_best_match_dam(
                typo_encoded, typo_len,
                self.dict_chars, self.dict_offsets, self.dict_lengths,
                cand_indices, num_cands)
        else:  # myers
            best_idx, best_dist = find_best_match_myers(
                typo_encoded, typo_len,
                self.dict_chars, self.dict_offsets, self.dict_lengths,
                cand_indices, num_cands)

        return (self.dictionary[cand_indices[best_idx]], int(best_dist))


def load_dictionary(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def run_benchmark(checker, typos, ground_truth, algorithm, algo_name, lang):
    """Run benchmark."""
    print(f"\n--- C Baseline (Numba JIT) {algo_name} + {lang} ({len(typos)} words) ---")
    sys.stdout.flush()

    # Warmup JIT
    print("  Warming up JIT...")
    _ = checker.find_correction(typos[0], algorithm)

    correct = 0
    total_time = 0

    for i, typo in enumerate(typos):
        start = time.perf_counter()
        best_word, best_dist = checker.find_correction(typo, algorithm)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if typo in ground_truth and best_word == ground_truth[typo]:
            correct += 1

        if (i + 1) % 50 == 0:
            avg_ms = total_time / (i + 1) * 1000
            print(f"  Progress: {i+1}/{len(typos)}, elapsed: {total_time:.1f}s, avg: {avg_ms:.1f}ms/word")
            sys.stdout.flush()

    ms_per_word = total_time / len(typos) * 1000
    accuracy = correct / len(typos) * 100

    print(f"  Completed: {len(typos)} words in {total_time:.1f}s, "
          f"{correct}/{len(typos)} correct ({accuracy:.1f}%), {ms_per_word:.1f}ms/word")
    sys.stdout.flush()

    return {
        'algorithm': algo_name,
        'language': lang,
        'num_words': len(typos),
        'total_time_s': total_time,
        'ms_per_word': ms_per_word,
        'correct': correct,
        'accuracy_pct': accuracy
    }


def main():
    print("=" * 80)
    print("C BASELINE (Numba JIT) BENCHMARK (500 words)")
    print("=" * 80)

    base = Path(__file__).parent.parent

    # Load data
    print("\nLoading dictionaries...")
    dict_mk = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
    dict_en = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
    print(f"  MK: {len(dict_mk):,} words, EN: {len(dict_en):,} words")

    # Load ground truth
    with open(base / "data/ground_truth/mk_hunspell_corrections_500.json", 'r', encoding='utf-8') as f:
        gt_mk = json.load(f)
    with open(base / "data/ground_truth/en_hunspell_corrections_500.json", 'r', encoding='utf-8') as f:
        gt_en = json.load(f)

    typos_mk = list(gt_mk.keys())
    typos_en = list(gt_en.keys())
    print(f"  MK typos: {len(typos_mk)}, EN typos: {len(typos_en)}")

    # Initialize checkers
    print("\nInitializing C baseline (pre-encoding dictionaries)...")
    checker_mk = CBaselineChecker(dict_mk)
    checker_en = CBaselineChecker(dict_en)
    print("  Done")
    sys.stdout.flush()

    # Python sequential baselines for comparison
    python_baseline = {
        ('Levenshtein', 'MK'): 351.7,
        ('Levenshtein', 'EN'): 319.9,
        ('Damerau-Levenshtein', 'MK'): 491.5,
        ('Damerau-Levenshtein', 'EN'): 459.2,
        ('Myers Bit-Vector', 'MK'): 94.9,
        ('Myers Bit-Vector', 'EN'): 82.1,
    }

    # CUDA baselines
    cuda_baseline = {
        ('Levenshtein', 'MK'): 2.89,
        ('Levenshtein', 'EN'): 1.50,
        ('Damerau-Levenshtein', 'MK'): 1.94,
        ('Damerau-Levenshtein', 'EN'): 1.81,
        ('Myers Bit-Vector', 'MK'): 1.41,
        ('Myers Bit-Vector', 'EN'): 1.20,
    }

    algorithms = [
        ('levenshtein', 'Levenshtein'),
        ('damerau', 'Damerau-Levenshtein'),
        ('myers', 'Myers Bit-Vector'),
    ]

    results = []

    for algo_key, algo_name in algorithms:
        r = run_benchmark(checker_mk, typos_mk, gt_mk, algo_key, algo_name, 'MK')
        results.append(r)
        r = run_benchmark(checker_en, typos_en, gt_en, algo_key, algo_name, 'EN')
        results.append(r)

    # Print comparison
    print("\n" + "=" * 120)
    print("C BASELINE vs PYTHON vs CUDA COMPARISON (500 words)")
    print("=" * 120)
    print(f"{'Algorithm':<20} | {'Lang':<4} | {'C/Numba (ms)':<12} | {'Python (ms)':<12} | {'CUDA (ms)':<10} | {'C vs Py':<10} | {'CUDA vs C':<10}")
    print("-" * 120)

    for r in results:
        c_ms = r['ms_per_word']
        py_ms = python_baseline.get((r['algorithm'], r['language']), 0)
        cuda_ms = cuda_baseline.get((r['algorithm'], r['language']), 0)

        c_vs_py = py_ms / c_ms if c_ms > 0 else 0
        cuda_vs_c = c_ms / cuda_ms if cuda_ms > 0 else 0

        print(f"{r['algorithm']:<20} | {r['language']:<4} | {c_ms:>10.1f}ms | {py_ms:>10.1f}ms | {cuda_ms:>8.2f}ms | {c_vs_py:>8.1f}x | {cuda_vs_c:>8.1f}x")

    print("-" * 120)

    # Save results
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        'description': 'C Baseline (Numba JIT) benchmark (500 words)',
        'results': results,
        'python_baseline': {f"{k[0]}_{k[1]}": v for k, v in python_baseline.items()},
        'cuda_baseline': {f"{k[0]}_{k[1]}": v for k, v in cuda_baseline.items()}
    }

    with open(results_dir / "c_baseline_500.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_dir / 'c_baseline_500.json'}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
