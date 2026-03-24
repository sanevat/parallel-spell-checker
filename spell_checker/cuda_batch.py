#!/usr/bin/env python3
"""
CUDA Batch Processing - Multiple typos per kernel launch.
Groups typos by length to share candidates, uses 2D grid for parallel processing.
Dictionary stays on GPU - only indices are passed per batch.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import cupy as cp

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Batch Levenshtein kernel - uses dict arrays + cand_indices
BATCH_LEVENSHTEIN_KERNEL = r'''
extern "C" __global__
void batch_levenshtein_kernel(
    const int* typos,           // All typos concatenated
    const int* typo_offsets,    // Start offset for each typo
    const int* typo_lengths,    // Length of each typo
    int num_typos,
    const int* dict_chars,      // Full dictionary chars (on GPU)
    const int* dict_offsets,    // Full dictionary offsets
    const int* dict_lengths,    // Full dictionary lengths
    const int* cand_indices,    // Candidate indices into dictionary
    int num_candidates,
    int* distances              // Output: [num_typos × num_candidates]
) {
    int typo_idx = blockIdx.y;
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (typo_idx >= num_typos || cand_idx >= num_candidates) return;

    // Get typo
    int t_start = typo_offsets[typo_idx];
    int t_len = typo_lengths[typo_idx];

    // Get candidate via index lookup
    int dict_idx = cand_indices[cand_idx];
    int c_start = dict_offsets[dict_idx];
    int c_len = dict_lengths[dict_idx];

    // Edge cases
    if (t_len == 0) {
        distances[typo_idx * num_candidates + cand_idx] = c_len;
        return;
    }
    if (c_len == 0) {
        distances[typo_idx * num_candidates + cand_idx] = t_len;
        return;
    }

    // DP with 2 rows
    int prev[64], curr[64];
    int max_len = c_len < 63 ? c_len : 63;

    for (int j = 0; j <= max_len; j++) prev[j] = j;

    for (int i = 1; i <= t_len; i++) {
        curr[0] = i;
        int t_char = typos[t_start + i - 1];

        for (int j = 1; j <= max_len; j++) {
            int cost = (t_char == dict_chars[c_start + j - 1]) ? 0 : 1;
            int ins = curr[j-1] + 1;
            int del = prev[j] + 1;
            int sub = prev[j-1] + cost;
            curr[j] = ins < del ? (ins < sub ? ins : sub) : (del < sub ? del : sub);
        }

        for (int j = 0; j <= max_len; j++) {
            int tmp = prev[j];
            prev[j] = curr[j];
            curr[j] = tmp;
        }
    }

    distances[typo_idx * num_candidates + cand_idx] = prev[max_len];
}
'''

# Batch Damerau-Levenshtein kernel
BATCH_DAMERAU_KERNEL = r'''
extern "C" __global__
void batch_damerau_kernel(
    const int* typos,
    const int* typo_offsets,
    const int* typo_lengths,
    int num_typos,
    const int* dict_chars,
    const int* dict_offsets,
    const int* dict_lengths,
    const int* cand_indices,
    int num_candidates,
    int* distances
) {
    int typo_idx = blockIdx.y;
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (typo_idx >= num_typos || cand_idx >= num_candidates) return;

    int t_start = typo_offsets[typo_idx];
    int t_len = typo_lengths[typo_idx];

    int dict_idx = cand_indices[cand_idx];
    int c_start = dict_offsets[dict_idx];
    int c_len = dict_lengths[dict_idx];

    if (t_len == 0) {
        distances[typo_idx * num_candidates + cand_idx] = c_len;
        return;
    }
    if (c_len == 0) {
        distances[typo_idx * num_candidates + cand_idx] = t_len;
        return;
    }

    int prev2[64], prev[64], curr[64];
    int max_len = c_len < 63 ? c_len : 63;

    // Initialize both prev2 and prev
    for (int j = 0; j <= max_len; j++) {
        prev2[j] = j;
        prev[j] = j;
    }

    for (int i = 1; i <= t_len; i++) {
        curr[0] = i;
        int t_char = typos[t_start + i - 1];
        int t_char_prev = (i > 1) ? typos[t_start + i - 2] : -1;

        for (int j = 1; j <= max_len; j++) {
            int c_char = dict_chars[c_start + j - 1];
            int cost = (t_char == c_char) ? 0 : 1;

            int ins = curr[j-1] + 1;
            int del = prev[j] + 1;
            int sub = prev[j-1] + cost;
            int min_cost = ins < del ? (ins < sub ? ins : sub) : (del < sub ? del : sub);

            // Transposition
            if (i > 1 && j > 1) {
                int c_char_prev = dict_chars[c_start + j - 2];
                if (t_char == c_char_prev && t_char_prev == c_char) {
                    int trans = prev2[j-2] + cost;
                    if (trans < min_cost) min_cost = trans;
                }
            }

            curr[j] = min_cost;
        }

        for (int j = 0; j <= max_len; j++) {
            prev2[j] = prev[j];
            prev[j] = curr[j];
        }
    }

    distances[typo_idx * num_candidates + cand_idx] = prev[max_len];
}
'''

# Batch Myers kernel with pre-built Peq
BATCH_MYERS_KERNEL = r'''
extern "C" __global__
void batch_myers_kernel(
    const int* typos,
    const int* typo_offsets,
    const int* typo_lengths,
    int num_typos,
    const int* dict_chars,
    const int* dict_offsets,
    const int* dict_lengths,
    const int* cand_indices,
    int num_candidates,
    const unsigned long long* peq_all,  // Pre-built Peq [num_typos × 256]
    int* distances
) {
    int typo_idx = blockIdx.y;
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (typo_idx >= num_typos || cand_idx >= num_candidates) return;

    int t_len = typo_lengths[typo_idx];

    int dict_idx = cand_indices[cand_idx];
    int c_start = dict_offsets[dict_idx];
    int c_len = dict_lengths[dict_idx];

    if (t_len == 0) {
        distances[typo_idx * num_candidates + cand_idx] = c_len;
        return;
    }
    if (c_len == 0) {
        distances[typo_idx * num_candidates + cand_idx] = t_len;
        return;
    }

    // For words > 64 chars, fall back to simple DP
    if (t_len > 64) {
        int t_start = typo_offsets[typo_idx];
        int prev[64], curr[64];
        int max_len = c_len < 63 ? c_len : 63;

        for (int j = 0; j <= max_len; j++) prev[j] = j;

        for (int i = 1; i <= t_len && i <= 64; i++) {
            curr[0] = i;
            int t_char = typos[t_start + i - 1];
            for (int j = 1; j <= max_len; j++) {
                int cost = (t_char == dict_chars[c_start + j - 1]) ? 0 : 1;
                int ins = curr[j-1] + 1;
                int del = prev[j] + 1;
                int sub = prev[j-1] + cost;
                curr[j] = ins < del ? (ins < sub ? ins : sub) : (del < sub ? del : sub);
            }
            for (int j = 0; j <= max_len; j++) {
                int tmp = prev[j]; prev[j] = curr[j]; curr[j] = tmp;
            }
        }
        distances[typo_idx * num_candidates + cand_idx] = prev[max_len];
        return;
    }

    // Get Peq for this typo
    const unsigned long long* Peq = peq_all + typo_idx * 256;

    // Myers algorithm
    unsigned long long Pv = 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long Mv = 0ULL;
    int score = t_len;
    unsigned long long last_bit = 1ULL << (t_len - 1);

    for (int j = 0; j < c_len; j++) {
        int c = dict_chars[c_start + j] & 0xFF;
        unsigned long long Eq = Peq[c];

        unsigned long long Xv = Eq | Mv;
        unsigned long long Xh = ((Eq & Pv) + Pv) ^ Pv | Eq;

        unsigned long long Ph = Mv | ~(Xh | Pv);
        unsigned long long Mh = Pv & Xh;

        if (Ph & last_bit) score++;
        if (Mh & last_bit) score--;

        Ph = (Ph << 1ULL) | 1ULL;
        Mh = Mh << 1ULL;

        Pv = Mh | ~(Xv | Ph);
        Mv = Ph & Xv;
    }

    distances[typo_idx * num_candidates + cand_idx] = score;
}
'''

# Argmin kernel - find best match per typo
ARGMIN_KERNEL = r'''
extern "C" __global__
void argmin_kernel(
    const int* distances,
    int num_typos,
    int num_candidates,
    int* best_indices,
    int* best_distances
) {
    int typo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (typo_idx >= num_typos) return;

    int best_idx = 0;
    int best_dist = distances[typo_idx * num_candidates];

    for (int i = 1; i < num_candidates; i++) {
        int dist = distances[typo_idx * num_candidates + i];
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    best_indices[typo_idx] = best_idx;
    best_distances[typo_idx] = best_dist;
}
'''


class CUDABatchChecker:
    """CUDA Batch spell checker - dictionary stays on GPU."""

    def __init__(self, dictionary: List[str]):
        """Pre-encode and transfer dictionary to GPU ONCE."""
        # Compile kernels
        self.lev_kernel = cp.RawKernel(BATCH_LEVENSHTEIN_KERNEL, 'batch_levenshtein_kernel')
        self.dam_kernel = cp.RawKernel(BATCH_DAMERAU_KERNEL, 'batch_damerau_kernel')
        self.myers_kernel = cp.RawKernel(BATCH_MYERS_KERNEL, 'batch_myers_kernel')
        self.argmin_kernel = cp.RawKernel(ARGMIN_KERNEL, 'argmin_kernel')

        # Encode dictionary
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

        # Dictionary on GPU (stays there)
        self.dict_chars_gpu = cp.array(all_chars, dtype=cp.int32)
        self.dict_offsets_gpu = cp.array(offsets, dtype=cp.int32)
        self.dict_lengths_gpu = cp.array(lengths, dtype=cp.int32)

        # Group dictionary by length for candidate filtering
        self.by_length = defaultdict(list)
        for i, word in enumerate(dictionary):
            self.by_length[len(word)].append(i)

        # Pre-build candidate indices arrays for common lengths (cache)
        self._cand_cache = {}

    def _get_cand_indices_gpu(self, typo_len: int, tol: int = 2) -> cp.ndarray:
        """Get candidate indices on GPU for a given typo length."""
        if typo_len in self._cand_cache:
            return self._cand_cache[typo_len]

        indices = []
        for length in range(max(1, typo_len - tol), typo_len + tol + 1):
            if length in self.by_length:
                indices.extend(self.by_length[length])

        if not indices:
            return None

        cand_gpu = cp.array(indices, dtype=cp.int32)
        self._cand_cache[typo_len] = cand_gpu
        return cand_gpu

    def _build_peq_batch(self, typos_encoded: List[List[int]], typo_lengths: List[int]) -> cp.ndarray:
        """Build Peq arrays for all typos in batch."""
        num_typos = len(typos_encoded)
        peq = np.zeros((num_typos, 256), dtype=np.uint64)

        for t_idx, (typo, t_len) in enumerate(zip(typos_encoded, typo_lengths)):
            for i in range(min(t_len, 64)):
                c = typo[i] & 0xFF
                peq[t_idx, c] |= np.uint64(1) << np.uint64(i)

        return cp.array(peq, dtype=cp.uint64)

    def find_corrections_batch(self, typos: List[str], algorithm: str = 'levenshtein') -> List[Tuple[str, int]]:
        """Find corrections for multiple typos using batch processing."""
        if not typos:
            return []

        # Group typos by length to share candidates
        length_groups = defaultdict(list)  # base_length -> [(idx, typo, encoded)]
        for idx, typo in enumerate(typos):
            base_len = len(typo)
            encoded = [ord(c) for c in typo]
            length_groups[base_len].append((idx, typo, encoded))

        results = [None] * len(typos)

        for base_len, group in length_groups.items():
            # Get candidate indices on GPU
            cand_indices_gpu = self._get_cand_indices_gpu(base_len)
            if cand_indices_gpu is None:
                for idx, typo, _ in group:
                    results[idx] = ("", 999999)
                continue

            num_candidates = len(cand_indices_gpu)
            num_typos_in_group = len(group)

            # Build typo arrays
            all_typo_chars = []
            typo_offsets = []
            typo_lengths = []
            typos_encoded = []
            offset = 0

            for idx, typo, encoded in group:
                all_typo_chars.extend(encoded)
                typo_offsets.append(offset)
                typo_lengths.append(len(typo))
                typos_encoded.append(encoded)
                offset += len(typo)

            # Transfer typos to GPU (only thing transferred per batch)
            typos_gpu = cp.array(all_typo_chars, dtype=cp.int32)
            typo_offsets_gpu = cp.array(typo_offsets, dtype=cp.int32)
            typo_lengths_gpu = cp.array(typo_lengths, dtype=cp.int32)

            # Allocate output
            distances_gpu = cp.zeros(num_typos_in_group * num_candidates, dtype=cp.int32)

            # Launch kernel with 2D grid
            threads_per_block = 256
            blocks_x = (num_candidates + threads_per_block - 1) // threads_per_block
            blocks_y = num_typos_in_group

            if algorithm == 'levenshtein':
                self.lev_kernel(
                    (blocks_x, blocks_y), (threads_per_block,),
                    (typos_gpu, typo_offsets_gpu, typo_lengths_gpu, num_typos_in_group,
                     self.dict_chars_gpu, self.dict_offsets_gpu, self.dict_lengths_gpu,
                     cand_indices_gpu, num_candidates, distances_gpu)
                )
            elif algorithm == 'damerau':
                self.dam_kernel(
                    (blocks_x, blocks_y), (threads_per_block,),
                    (typos_gpu, typo_offsets_gpu, typo_lengths_gpu, num_typos_in_group,
                     self.dict_chars_gpu, self.dict_offsets_gpu, self.dict_lengths_gpu,
                     cand_indices_gpu, num_candidates, distances_gpu)
                )
            else:  # myers
                peq_gpu = self._build_peq_batch(typos_encoded, typo_lengths)
                self.myers_kernel(
                    (blocks_x, blocks_y), (threads_per_block,),
                    (typos_gpu, typo_offsets_gpu, typo_lengths_gpu, num_typos_in_group,
                     self.dict_chars_gpu, self.dict_offsets_gpu, self.dict_lengths_gpu,
                     cand_indices_gpu, num_candidates, peq_gpu, distances_gpu)
                )

            # Find argmin per row on GPU
            best_indices_gpu = cp.zeros(num_typos_in_group, dtype=cp.int32)
            best_distances_gpu = cp.zeros(num_typos_in_group, dtype=cp.int32)

            argmin_threads = 256
            argmin_blocks = (num_typos_in_group + argmin_threads - 1) // argmin_threads

            self.argmin_kernel(
                (argmin_blocks,), (argmin_threads,),
                (distances_gpu, num_typos_in_group, num_candidates,
                 best_indices_gpu, best_distances_gpu)
            )

            # Get results
            best_indices = best_indices_gpu.get()
            best_distances = best_distances_gpu.get()
            cand_indices_cpu = cand_indices_gpu.get()

            for i, (orig_idx, typo, _) in enumerate(group):
                best_cand_idx = cand_indices_cpu[best_indices[i]]
                results[orig_idx] = (self.dictionary[best_cand_idx], int(best_distances[i]))

        return results


def load_dictionary(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def run_benchmark(checker, typos, ground_truth, algorithm, algo_name, lang, batch_size=64):
    """Run batch benchmark."""
    print(f"\n--- CUDA Batch {algo_name} + {lang} ({len(typos)} words, batch={batch_size}) ---")
    sys.stdout.flush()

    correct = 0
    total_time = 0

    # Process in batches
    for batch_start in range(0, len(typos), batch_size):
        batch_end = min(batch_start + batch_size, len(typos))
        batch_typos = typos[batch_start:batch_end]

        start = time.perf_counter()
        results = checker.find_corrections_batch(batch_typos, algorithm)
        cp.cuda.Stream.null.synchronize()  # Ensure kernel completes
        elapsed = time.perf_counter() - start
        total_time += elapsed

        for i, (best_word, best_dist) in enumerate(results):
            typo = batch_typos[i]
            if typo in ground_truth and best_word == ground_truth[typo]:
                correct += 1

        if (batch_end) % 100 == 0 or batch_end == len(typos):
            avg_ms = total_time / batch_end * 1000
            print(f"  Progress: {batch_end}/{len(typos)}, elapsed: {total_time:.2f}s, avg: {avg_ms:.3f}ms/word")
            sys.stdout.flush()

    ms_per_word = total_time / len(typos) * 1000
    accuracy = correct / len(typos) * 100

    print(f"  Completed: {len(typos)} words in {total_time:.2f}s, "
          f"{correct}/{len(typos)} correct ({accuracy:.1f}%), {ms_per_word:.3f}ms/word")
    sys.stdout.flush()

    return {
        'algorithm': algo_name,
        'language': lang,
        'num_words': len(typos),
        'batch_size': batch_size,
        'total_time_s': total_time,
        'ms_per_word': ms_per_word,
        'correct': correct,
        'accuracy_pct': accuracy
    }


def main():
    print("=" * 100)
    print("CUDA BATCH PROCESSING BENCHMARK (500 words)")
    print("=" * 100)

    base = Path(__file__).parent.parent

    # Check CUDA
    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"\nCUDA device: {device_name}")

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

    # Initialize batch checkers
    print("\nInitializing CUDA Batch checkers (dictionary on GPU)...")
    checker_mk = CUDABatchChecker(dict_mk)
    checker_en = CUDABatchChecker(dict_en)
    print("  Done")
    sys.stdout.flush()

    # Warmup
    print("\nWarming up kernels...")
    _ = checker_mk.find_corrections_batch(typos_mk[:10], 'levenshtein')
    _ = checker_mk.find_corrections_batch(typos_mk[:10], 'damerau')
    _ = checker_mk.find_corrections_batch(typos_mk[:10], 'myers')
    cp.cuda.Stream.null.synchronize()
    print("  Done")
    sys.stdout.flush()

    # Baselines for comparison
    per_word_cuda = {
        ('Levenshtein', 'MK'): 2.89,
        ('Levenshtein', 'EN'): 1.50,
        ('Damerau-Levenshtein', 'MK'): 1.94,
        ('Damerau-Levenshtein', 'EN'): 1.81,
        ('Myers Bit-Vector', 'MK'): 1.41,
        ('Myers Bit-Vector', 'EN'): 1.20,
    }

    c_baseline = {
        ('Levenshtein', 'MK'): 6.49,
        ('Levenshtein', 'EN'): 6.59,
        ('Damerau-Levenshtein', 'MK'): 9.10,
        ('Damerau-Levenshtein', 'EN'): 8.70,
        ('Myers Bit-Vector', 'MK'): 3.67,
        ('Myers Bit-Vector', 'EN'): 3.89,
    }

    algorithms = [
        ('levenshtein', 'Levenshtein'),
        ('damerau', 'Damerau-Levenshtein'),
        ('myers', 'Myers Bit-Vector'),
    ]

    results = []

    for algo_key, algo_name in algorithms:
        r = run_benchmark(checker_mk, typos_mk, gt_mk, algo_key, algo_name, 'MK', batch_size=64)
        results.append(r)
        r = run_benchmark(checker_en, typos_en, gt_en, algo_key, algo_name, 'EN', batch_size=64)
        results.append(r)

    # Print comparison
    print("\n" + "=" * 130)
    print("CUDA BATCH vs PER-WORD vs C BASELINE COMPARISON (500 words)")
    print("=" * 130)
    print(f"{'Algorithm':<20} | {'Lang':<4} | {'Per-word CUDA':<14} | {'Batch CUDA':<12} | {'C Baseline':<12} | {'Batch vs Per-word':<18} | {'Batch vs C':<12}")
    print("-" * 130)

    for r in results:
        batch_ms = r['ms_per_word']
        per_word_ms = per_word_cuda.get((r['algorithm'], r['language']), 0)
        c_ms = c_baseline.get((r['algorithm'], r['language']), 0)

        batch_vs_perword = per_word_ms / batch_ms if batch_ms > 0 else 0
        batch_vs_c = c_ms / batch_ms if batch_ms > 0 else 0

        print(f"{r['algorithm']:<20} | {r['language']:<4} | {per_word_ms:>10.2f}ms   | {batch_ms:>8.3f}ms   | {c_ms:>8.2f}ms   | {batch_vs_perword:>14.1f}x   | {batch_vs_c:>8.1f}x")

    print("-" * 130)

    # Save results
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        'description': 'CUDA Batch Processing benchmark (500 words)',
        'device': device_name,
        'batch_size': 64,
        'results': results,
        'per_word_cuda': {f"{k[0]}_{k[1]}": v for k, v in per_word_cuda.items()},
        'c_baseline': {f"{k[0]}_{k[1]}": v for k, v in c_baseline.items()}
    }

    with open(results_dir / "cuda_batch_500.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_dir / 'cuda_batch_500.json'}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
