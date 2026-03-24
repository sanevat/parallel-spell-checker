#!/usr/bin/env python3
"""
CUDA Global Memory Spell Checker using CuPy.
- Pre-transfers dictionary to GPU ONCE
- Each thread computes edit distance for 1 candidate
- Real Myers bit-vector implementation for GPU
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cupy as cp

# CUDA kernel for Levenshtein distance
LEVENSHTEIN_KERNEL = r'''
extern "C" __global__ void levenshtein_kernel(
    const int* typo,
    int typo_len,
    const int* dict_chars,
    const int* dict_offsets,
    const int* dict_lengths,
    const int* cand_indices,
    int num_candidates,
    int* distances
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    int cand_idx = cand_indices[idx];
    int cand_start = dict_offsets[cand_idx];
    int cand_len = dict_lengths[cand_idx];

    int prev[65];
    int curr[65];

    for (int j = 0; j <= cand_len; j++) {
        prev[j] = j;
    }

    for (int i = 1; i <= typo_len; i++) {
        curr[0] = i;
        int typo_char = typo[i - 1];

        for (int j = 1; j <= cand_len; j++) {
            int cand_char = dict_chars[cand_start + j - 1];
            int cost = (typo_char == cand_char) ? 0 : 1;

            int insert_cost = curr[j - 1] + 1;
            int delete_cost = prev[j] + 1;
            int replace_cost = prev[j - 1] + cost;

            int min_cost = insert_cost;
            if (delete_cost < min_cost) min_cost = delete_cost;
            if (replace_cost < min_cost) min_cost = replace_cost;

            curr[j] = min_cost;
        }

        for (int j = 0; j <= cand_len; j++) {
            prev[j] = curr[j];
        }
    }

    distances[idx] = prev[cand_len];
}
'''

# CUDA kernel for Damerau-Levenshtein distance
DAMERAU_KERNEL = r'''
extern "C" __global__ void damerau_kernel(
    const int* typo,
    int typo_len,
    const int* dict_chars,
    const int* dict_offsets,
    const int* dict_lengths,
    const int* cand_indices,
    int num_candidates,
    int* distances
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    int cand_idx = cand_indices[idx];
    int cand_start = dict_offsets[cand_idx];
    int cand_len = dict_lengths[cand_idx];

    int prev2[65];
    int prev[65];
    int curr[65];

    for (int j = 0; j <= cand_len; j++) {
        prev2[j] = j > 0 ? j - 1 : 0;
        prev[j] = j;
    }

    for (int i = 1; i <= typo_len; i++) {
        curr[0] = i;
        int typo_char = typo[i - 1];
        int typo_char_prev = (i > 1) ? typo[i - 2] : -1;

        for (int j = 1; j <= cand_len; j++) {
            int cand_char = dict_chars[cand_start + j - 1];
            int cand_char_prev = (j > 1) ? dict_chars[cand_start + j - 2] : -1;
            int cost = (typo_char == cand_char) ? 0 : 1;

            int insert_cost = curr[j - 1] + 1;
            int delete_cost = prev[j] + 1;
            int replace_cost = prev[j - 1] + cost;

            int min_cost = insert_cost;
            if (delete_cost < min_cost) min_cost = delete_cost;
            if (replace_cost < min_cost) min_cost = replace_cost;

            if (i > 1 && j > 1 && typo_char == cand_char_prev && typo_char_prev == cand_char) {
                int trans_cost = prev2[j - 2] + cost;
                if (trans_cost < min_cost) min_cost = trans_cost;
            }

            curr[j] = min_cost;
        }

        for (int j = 0; j <= cand_len; j++) {
            prev2[j] = prev[j];
            prev[j] = curr[j];
        }
    }

    distances[idx] = prev[cand_len];
}
'''

# Real Myers bit-vector kernel
# Uses 64-bit vectors, works for words up to 64 chars
MYERS_KERNEL = r'''
extern "C" __global__ void myers_kernel(
    const int* typo,
    int typo_len,
    const unsigned long long* typo_peq,  // Peq masks for typo chars (256 entries)
    const int* dict_chars,
    const int* dict_offsets,
    const int* dict_lengths,
    const int* cand_indices,
    int num_candidates,
    int* distances
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    int cand_idx = cand_indices[idx];
    int cand_start = dict_offsets[cand_idx];
    int cand_len = dict_lengths[cand_idx];

    if (typo_len == 0) {
        distances[idx] = cand_len;
        return;
    }
    if (cand_len == 0) {
        distances[idx] = typo_len;
        return;
    }

    // Myers bit-vector algorithm
    unsigned long long Pv = ~0ULL;
    unsigned long long Mv = 0ULL;
    int score = typo_len;

    unsigned long long last_bit = 1ULL << (typo_len - 1);

    for (int j = 0; j < cand_len; j++) {
        int c = dict_chars[cand_start + j];
        // Get Peq for this character (use modulo for safety with large Unicode)
        unsigned long long Eq = typo_peq[c & 0xFFFF];

        unsigned long long Xv = Eq | Mv;
        unsigned long long Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;

        unsigned long long Ph = Mv | ~(Xh | Pv);
        unsigned long long Mh = Pv & Xh;

        if (Ph & last_bit) score++;
        if (Mh & last_bit) score--;

        Ph = (Ph << 1) | 1ULL;
        Mh = Mh << 1;

        Pv = Mh | ~(Xv | Ph);
        Mv = Ph & Xv;
    }

    distances[idx] = score;
}
'''


class CUDASpellChecker:
    def __init__(self, dictionary: List[str]):
        """Initialize with dictionary pre-loaded to GPU."""
        # Compile kernels
        self.levenshtein_kernel = cp.RawKernel(LEVENSHTEIN_KERNEL, 'levenshtein_kernel')
        self.damerau_kernel = cp.RawKernel(DAMERAU_KERNEL, 'damerau_kernel')
        self.myers_kernel = cp.RawKernel(MYERS_KERNEL, 'myers_kernel')
        self.block_size = 256

        # Encode and transfer dictionary to GPU ONCE
        self._setup_dictionary(dictionary)

    def _setup_dictionary(self, dictionary: List[str]):
        """Encode dictionary and transfer to GPU."""
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

        # Store on CPU for candidate lookup
        self.dictionary = dictionary
        self.word_lengths = np.array(lengths, dtype=np.int32)

        # Transfer to GPU (ONCE)
        self.d_dict_chars = cp.asarray(np.array(all_chars, dtype=np.int32))
        self.d_dict_offsets = cp.asarray(np.array(offsets, dtype=np.int32))
        self.d_dict_lengths = cp.asarray(self.word_lengths)

        # Group words by length for fast candidate lookup
        self.by_length = defaultdict(list)
        for i, word in enumerate(dictionary):
            self.by_length[len(word)].append(i)

    def _get_candidate_indices(self, typo_len: int, tol: int = 2) -> np.ndarray:
        """Get indices of candidates within length tolerance."""
        indices = []
        for length in range(max(1, typo_len - tol), typo_len + tol + 1):
            if length in self.by_length:
                indices.extend(self.by_length[length])
        return np.array(indices, dtype=np.int32)

    def _build_peq(self, typo: str) -> np.ndarray:
        """Build Peq masks for Myers algorithm."""
        # Use 65536 entries to handle Unicode (BMP)
        peq = np.zeros(65536, dtype=np.uint64)
        for i, c in enumerate(typo):
            peq[ord(c) & 0xFFFF] |= (1 << i)
        return peq

    def find_correction(self, typo: str, algorithm: str = 'levenshtein') -> Tuple[str, int]:
        """Find best correction using CUDA."""
        typo_len = len(typo)
        cand_indices = self._get_candidate_indices(typo_len)

        if len(cand_indices) == 0:
            return ("", 999999)

        num_candidates = len(cand_indices)

        # Transfer only typo and candidate indices (small)
        typo_encoded = np.array([ord(c) for c in typo], dtype=np.int32)
        d_typo = cp.asarray(typo_encoded)
        d_cand_indices = cp.asarray(cand_indices)
        d_distances = cp.zeros(num_candidates, dtype=np.int32)

        grid_size = (num_candidates + self.block_size - 1) // self.block_size

        if algorithm == 'levenshtein':
            self.levenshtein_kernel(
                (grid_size,), (self.block_size,),
                (d_typo, typo_len, self.d_dict_chars, self.d_dict_offsets,
                 self.d_dict_lengths, d_cand_indices, num_candidates, d_distances)
            )
        elif algorithm == 'damerau':
            self.damerau_kernel(
                (grid_size,), (self.block_size,),
                (d_typo, typo_len, self.d_dict_chars, self.d_dict_offsets,
                 self.d_dict_lengths, d_cand_indices, num_candidates, d_distances)
            )
        else:  # myers
            # Build Peq on CPU and transfer
            peq = self._build_peq(typo)
            d_peq = cp.asarray(peq)
            self.myers_kernel(
                (grid_size,), (self.block_size,),
                (d_typo, typo_len, d_peq, self.d_dict_chars, self.d_dict_offsets,
                 self.d_dict_lengths, d_cand_indices, num_candidates, d_distances)
            )

        # Get results
        distances = cp.asnumpy(d_distances)
        min_idx = np.argmin(distances)
        best_word_idx = cand_indices[min_idx]

        return (self.dictionary[best_word_idx], int(distances[min_idx]))


def load_dictionary(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def run_benchmark(checker, typos, ground_truth, algorithm, algo_name, lang):
    """Run CUDA benchmark."""
    print(f"\n--- CUDA Global {algo_name} + {lang} ({len(typos)} words) ---")
    sys.stdout.flush()

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
    print("CUDA GLOBAL MEMORY SPELL CHECKER (500 words)")
    print("=" * 80)

    base = Path(__file__).parent.parent

    # Check CUDA
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"\nCUDA device: {device_name}")
    print(f"Compute capability: {device.compute_capability}")

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

    # Initialize CUDA checkers (transfers dict to GPU)
    print("\nInitializing CUDA (transferring dictionaries to GPU)...")
    checker_mk = CUDASpellChecker(dict_mk)
    checker_en = CUDASpellChecker(dict_en)
    print("  Done")
    sys.stdout.flush()

    # Sequential baselines
    seq_baseline = {
        ('Levenshtein', 'MK'): 351.7,
        ('Levenshtein', 'EN'): 319.9,
        ('Damerau-Levenshtein', 'MK'): 491.5,
        ('Damerau-Levenshtein', 'EN'): 459.2,
        ('Myers Bit-Vector', 'MK'): 94.9,
        ('Myers Bit-Vector', 'EN'): 82.1,
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
    print("\n" + "=" * 100)
    print("CUDA GLOBAL vs SEQUENTIAL COMPARISON (500 words)")
    print("=" * 100)
    print(f"{'Algorithm':<20} | {'Lang':<4} | {'CUDA (ms)':<12} | {'Sequential (ms)':<15} | {'Speedup':<10}")
    print("-" * 100)

    for r in results:
        seq_ms = seq_baseline.get((r['algorithm'], r['language']), 0)
        cuda_ms = r['ms_per_word']
        speedup = seq_ms / cuda_ms if cuda_ms > 0 else 0
        print(f"{r['algorithm']:<20} | {r['language']:<4} | {cuda_ms:>10.1f}ms | {seq_ms:>13.1f}ms | {speedup:>8.2f}x")

    print("-" * 100)

    # Save results
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        'description': 'CUDA Global Memory benchmark (500 words)',
        'device': device_name,
        'results': results,
        'sequential_baseline': {f"{k[0]}_{k[1]}": v for k, v in seq_baseline.items()}
    }

    with open(results_dir / "cuda_global_500.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_dir / 'cuda_global_500.json'}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
