#!/usr/bin/env python3
"""
CUDA Batch Processing with Shared Memory Optimization.
Compares: CUDA Shared Memory vs CUDA Global Memory vs C Sequential
"""

import os
import sys
import json
import time
import statistics
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict

NUM_RUNS = 4  # Number of runs for median calculation

# Allow running directly: python spell_checker/cuda_batch_shared.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Shared Memory Levenshtein kernel
SHARED_LEVENSHTEIN_KERNEL = r'''
extern "C" __global__
void shared_levenshtein_kernel(
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
    __shared__ int s_typo[64];
    __shared__ int s_typo_len;

    int typo_idx = blockIdx.y;
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (typo_idx >= num_typos) return;

    if (threadIdx.x == 0) {
        s_typo_len = typo_lengths[typo_idx];
        int t_start = typo_offsets[typo_idx];
        int len = s_typo_len < 64 ? s_typo_len : 64;
        for (int i = 0; i < len; i++) {
            s_typo[i] = typos[t_start + i];
        }
    }
    __syncthreads();

    if (cand_idx >= num_candidates) return;

    int t_len = s_typo_len;
    int dict_idx = cand_indices[cand_idx];
    int c_start = dict_offsets[dict_idx];
    int c_len = dict_lengths[dict_idx];

    if (t_len == 0) { distances[typo_idx * num_candidates + cand_idx] = c_len; return; }
    if (c_len == 0) { distances[typo_idx * num_candidates + cand_idx] = t_len; return; }

    int prev[64], curr[64];
    int max_len = c_len < 63 ? c_len : 63;

    for (int j = 0; j <= max_len; j++) prev[j] = j;

    for (int i = 1; i <= t_len; i++) {
        curr[0] = i;
        int t_char = s_typo[i - 1];
        for (int j = 1; j <= max_len; j++) {
            int cost = (t_char == dict_chars[c_start + j - 1]) ? 0 : 1;
            int ins = curr[j-1] + 1;
            int del = prev[j] + 1;
            int sub = prev[j-1] + cost;
            curr[j] = ins < del ? (ins < sub ? ins : sub) : (del < sub ? del : sub);
        }
        for (int j = 0; j <= max_len; j++) { int tmp = prev[j]; prev[j] = curr[j]; curr[j] = tmp; }
    }
    distances[typo_idx * num_candidates + cand_idx] = prev[max_len];
}
'''

# Shared Memory Damerau-Levenshtein kernel
SHARED_DAMERAU_KERNEL = r'''
extern "C" __global__
void shared_damerau_kernel(
    const int* typos, const int* typo_offsets, const int* typo_lengths, int num_typos,
    const int* dict_chars, const int* dict_offsets, const int* dict_lengths,
    const int* cand_indices, int num_candidates, int* distances
) {
    __shared__ int s_typo[64];
    __shared__ int s_typo_len;

    int typo_idx = blockIdx.y;
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (typo_idx >= num_typos) return;

    if (threadIdx.x == 0) {
        s_typo_len = typo_lengths[typo_idx];
        int t_start = typo_offsets[typo_idx];
        int len = s_typo_len < 64 ? s_typo_len : 64;
        for (int i = 0; i < len; i++) s_typo[i] = typos[t_start + i];
    }
    __syncthreads();

    if (cand_idx >= num_candidates) return;

    int t_len = s_typo_len;
    int dict_idx = cand_indices[cand_idx];
    int c_start = dict_offsets[dict_idx];
    int c_len = dict_lengths[dict_idx];

    if (t_len == 0) { distances[typo_idx * num_candidates + cand_idx] = c_len; return; }
    if (c_len == 0) { distances[typo_idx * num_candidates + cand_idx] = t_len; return; }

    int prev2[64], prev[64], curr[64];
    int max_len = c_len < 63 ? c_len : 63;

    for (int j = 0; j <= max_len; j++) { prev2[j] = j; prev[j] = j; }

    for (int i = 1; i <= t_len; i++) {
        curr[0] = i;
        int t_char = s_typo[i - 1];
        int t_char_prev = (i > 1) ? s_typo[i - 2] : -1;

        for (int j = 1; j <= max_len; j++) {
            int c_char = dict_chars[c_start + j - 1];
            int cost = (t_char == c_char) ? 0 : 1;
            int ins = curr[j-1] + 1, del = prev[j] + 1, sub = prev[j-1] + cost;
            int min_cost = ins < del ? (ins < sub ? ins : sub) : (del < sub ? del : sub);

            if (i > 1 && j > 1) {
                int c_char_prev = dict_chars[c_start + j - 2];
                if (t_char == c_char_prev && t_char_prev == c_char) {
                    int trans = prev2[j-2] + cost;
                    if (trans < min_cost) min_cost = trans;
                }
            }
            curr[j] = min_cost;
        }
        for (int j = 0; j <= max_len; j++) { prev2[j] = prev[j]; prev[j] = curr[j]; }
    }
    distances[typo_idx * num_candidates + cand_idx] = prev[max_len];
}
'''

# Shared Memory Myers kernel
SHARED_MYERS_KERNEL = r'''
extern "C" __global__
void shared_myers_kernel(
    const int* typos, const int* typo_offsets, const int* typo_lengths, int num_typos,
    const int* dict_chars, const int* dict_offsets, const int* dict_lengths,
    const int* cand_indices, int num_candidates,
    const unsigned long long* peq_all, int* distances
) {
    __shared__ unsigned long long s_peq[256];
    __shared__ int s_typo_len;

    int typo_idx = blockIdx.y;
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (typo_idx >= num_typos) return;

    const unsigned long long* Peq_src = peq_all + typo_idx * 256;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_peq[i] = Peq_src[i];
    if (threadIdx.x == 0) s_typo_len = typo_lengths[typo_idx];
    __syncthreads();

    if (cand_idx >= num_candidates) return;

    int t_len = s_typo_len;
    int dict_idx = cand_indices[cand_idx];
    int c_start = dict_offsets[dict_idx];
    int c_len = dict_lengths[dict_idx];

    if (t_len == 0) { distances[typo_idx * num_candidates + cand_idx] = c_len; return; }
    if (c_len == 0) { distances[typo_idx * num_candidates + cand_idx] = t_len; return; }

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
                int ins = curr[j-1] + 1, del = prev[j] + 1, sub = prev[j-1] + cost;
                curr[j] = ins < del ? (ins < sub ? ins : sub) : (del < sub ? del : sub);
            }
            for (int j = 0; j <= max_len; j++) { int tmp = prev[j]; prev[j] = curr[j]; curr[j] = tmp; }
        }
        distances[typo_idx * num_candidates + cand_idx] = prev[max_len];
        return;
    }

    unsigned long long Pv = 0xFFFFFFFFFFFFFFFFULL, Mv = 0ULL;
    int score = t_len;
    unsigned long long last_bit = 1ULL << (t_len - 1);

    for (int j = 0; j < c_len; j++) {
        int c = dict_chars[c_start + j] & 0xFF;
        unsigned long long Eq = s_peq[c];
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

ARGMIN_KERNEL = r'''
extern "C" __global__
void argmin_kernel(const int* distances, int num_typos, int num_candidates, int* best_indices, int* best_distances) {
    int typo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (typo_idx >= num_typos) return;
    int best_idx = 0, best_dist = distances[typo_idx * num_candidates];
    for (int i = 1; i < num_candidates; i++) {
        int dist = distances[typo_idx * num_candidates + i];
        if (dist < best_dist) { best_dist = dist; best_idx = i; }
    }
    best_indices[typo_idx] = best_idx;
    best_distances[typo_idx] = best_dist;
}
'''


class CUDABatchSharedChecker:
    """CUDA Batch spell checker with shared memory."""

    def __init__(self, dictionary: List[str]):
        self.lev_kernel = cp.RawKernel(SHARED_LEVENSHTEIN_KERNEL, 'shared_levenshtein_kernel')
        self.dam_kernel = cp.RawKernel(SHARED_DAMERAU_KERNEL, 'shared_damerau_kernel')
        self.myers_kernel = cp.RawKernel(SHARED_MYERS_KERNEL, 'shared_myers_kernel')
        self.argmin_kernel = cp.RawKernel(ARGMIN_KERNEL, 'argmin_kernel')

        all_chars, offsets, lengths = [], [], []
        offset = 0
        for word in dictionary:
            encoded = [ord(c) for c in word]
            all_chars.extend(encoded)
            offsets.append(offset)
            lengths.append(len(word))
            offset += len(word)

        self.dictionary = dictionary
        self.dict_chars_gpu = cp.array(all_chars, dtype=cp.int32)
        self.dict_offsets_gpu = cp.array(offsets, dtype=cp.int32)
        self.dict_lengths_gpu = cp.array(lengths, dtype=cp.int32)

        self.by_length = defaultdict(list)
        for i, word in enumerate(dictionary):
            self.by_length[len(word)].append(i)
        self._cand_cache = {}

    def _get_cand_indices_gpu(self, typo_len: int, tol: int = 2):
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

    def _build_peq_batch(self, typos_encoded, typo_lengths):
        num_typos = len(typos_encoded)
        peq = np.zeros((num_typos, 256), dtype=np.uint64)
        for t_idx, (typo, t_len) in enumerate(zip(typos_encoded, typo_lengths)):
            for i in range(min(t_len, 64)):
                c = typo[i] & 0xFF
                peq[t_idx, c] |= np.uint64(1) << np.uint64(i)
        return cp.array(peq, dtype=cp.uint64)

    def find_corrections_batch(self, typos: List[str], algorithm: str = 'levenshtein'):
        if not typos:
            return []

        length_groups = defaultdict(list)
        for idx, typo in enumerate(typos):
            encoded = [ord(c) for c in typo]
            length_groups[len(typo)].append((idx, typo, encoded))

        results = [None] * len(typos)

        for base_len, group in length_groups.items():
            cand_indices_gpu = self._get_cand_indices_gpu(base_len)
            if cand_indices_gpu is None:
                for idx, typo, _ in group:
                    results[idx] = ("", 999999)
                continue

            num_candidates = len(cand_indices_gpu)
            num_typos_in_group = len(group)

            all_typo_chars, typo_offsets, typo_lengths, typos_encoded = [], [], [], []
            offset = 0
            for idx, typo, encoded in group:
                all_typo_chars.extend(encoded)
                typo_offsets.append(offset)
                typo_lengths.append(len(typo))
                typos_encoded.append(encoded)
                offset += len(typo)

            typos_gpu = cp.array(all_typo_chars, dtype=cp.int32)
            typo_offsets_gpu = cp.array(typo_offsets, dtype=cp.int32)
            typo_lengths_gpu = cp.array(typo_lengths, dtype=cp.int32)
            distances_gpu = cp.zeros(num_typos_in_group * num_candidates, dtype=cp.int32)

            threads = 256
            blocks_x = (num_candidates + threads - 1) // threads
            blocks_y = num_typos_in_group

            if algorithm == 'levenshtein':
                self.lev_kernel((blocks_x, blocks_y), (threads,),
                    (typos_gpu, typo_offsets_gpu, typo_lengths_gpu, num_typos_in_group,
                     self.dict_chars_gpu, self.dict_offsets_gpu, self.dict_lengths_gpu,
                     cand_indices_gpu, num_candidates, distances_gpu))
            elif algorithm == 'damerau':
                self.dam_kernel((blocks_x, blocks_y), (threads,),
                    (typos_gpu, typo_offsets_gpu, typo_lengths_gpu, num_typos_in_group,
                     self.dict_chars_gpu, self.dict_offsets_gpu, self.dict_lengths_gpu,
                     cand_indices_gpu, num_candidates, distances_gpu))
            else:
                peq_gpu = self._build_peq_batch(typos_encoded, typo_lengths)
                self.myers_kernel((blocks_x, blocks_y), (threads,),
                    (typos_gpu, typo_offsets_gpu, typo_lengths_gpu, num_typos_in_group,
                     self.dict_chars_gpu, self.dict_offsets_gpu, self.dict_lengths_gpu,
                     cand_indices_gpu, num_candidates, peq_gpu, distances_gpu))

            best_indices_gpu = cp.zeros(num_typos_in_group, dtype=cp.int32)
            best_distances_gpu = cp.zeros(num_typos_in_group, dtype=cp.int32)
            argmin_blocks = (num_typos_in_group + 255) // 256
            self.argmin_kernel((argmin_blocks,), (256,),
                (distances_gpu, num_typos_in_group, num_candidates, best_indices_gpu, best_distances_gpu))

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


def load_existing_results(base: Path) -> Dict:
    """Load existing CUDA Global and C results from JSON file."""
    results_file = base / "results" / "cuda_batch_vs_c_comprehensive.json"
    if not results_file.exists():
        return {}

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build lookup: (algorithm, language, num_words) -> {cuda_ms, c_ms}
    lookup = {}
    for r in data.get('results', []):
        key = (r['algorithm'], r['language'], r['num_words'])
        if r['method'] == 'CUDA Batch':
            if key not in lookup:
                lookup[key] = {}
            lookup[key]['cuda_global_ms'] = r['ms_per_word']
        elif r['method'] == 'C-Sequential':
            if key not in lookup:
                lookup[key] = {}
            lookup[key]['c_ms'] = r['ms_per_word']

    return lookup


def main():
    """Compare CUDA Shared vs CUDA Global vs C Sequential."""
    print("=" * 110)
    print("CUDA BATCH SHARED MEMORY vs CUDA GLOBAL vs C SEQUENTIAL")
    print(f"(CUDA Global and C results loaded from existing file, Shared runs {NUM_RUNS}x with median)")
    print("=" * 110)

    base = Path(__file__).parent.parent

    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"\nCUDA device: {device_name}")

    # Load existing CUDA Global and C results
    existing_results = load_existing_results(base)
    if not existing_results:
        print("ERROR: No existing results found in results/cuda_batch_vs_c_comprehensive.json")
        print("Please run cuda_batch benchmark first.")
        return
    print(f"Loaded {len(existing_results)} existing result entries")

    # Load data
    languages = {
        'MK': 'data/dictionary/mk_equal.txt',
        'EN': 'data/dictionary/en_equal.txt',
        'TR': 'data/dictionary/tr_equal.txt',
    }

    # All test sizes - try multiple naming conventions
    size_variants = [
        ('250', ['250']),
        ('500', ['500']),
        ('1K', ['1K', '1000']),
        ('1500', ['1500']),
        ('2K', ['2K', '2000']),
        ('2500', ['2500']),
        ('3500', ['3500']),
        ('5K', ['5K', '5000']),
    ]

    algorithms = [('levenshtein', 'Levenshtein'), ('damerau', 'Damerau-Levenshtein'), ('myers', 'Myers Bit-Vector')]
    all_results = []

    for lang, dict_file in languages.items():
        dict_path = base / dict_file
        if not dict_path.exists():
            print(f"Dictionary not found: {dict_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Language: {lang}")
        print(f"{'='*70}")

        dictionary = load_dictionary(str(dict_path))
        print(f"Dictionary: {len(dictionary):,} words")

        # Initialize ONLY shared memory checker
        checker_shared = CUDABatchSharedChecker(dictionary)

        for size_label, variants in size_variants:
            # Try each variant until one exists
            gt_path = None
            for variant in variants:
                candidate = base / f"data/ground_truth/{lang.lower()}_hunspell_corrections_{variant}.json"
                if candidate.exists():
                    gt_path = candidate
                    break

            if gt_path is None:
                print(f"  Skipping size {size_label}: no file found")
                continue

            with open(gt_path, 'r', encoding='utf-8') as f:
                gt = json.load(f)
            typos = list(gt.keys())

            print(f"\n  --- Size: {size_label} ({len(typos)} typos) ---")

            # Warmup shared memory only
            for algo_key, _ in algorithms:
                _ = checker_shared.find_corrections_batch(typos[:5], algo_key)
            cp.cuda.Stream.null.synchronize()

            batch_size = 64

            for algo_key, algo_name in algorithms:
                # Look up existing CUDA Global and C results
                key = (algo_name, lang, len(typos))
                if key not in existing_results:
                    print(f"    {algo_name}: No existing results for {len(typos)} typos, skipping")
                    continue

                ms_global = existing_results[key].get('cuda_global_ms', 0)
                ms_c = existing_results[key].get('c_ms', 0)

                # CUDA Shared - run NUM_RUNS times and take median
                run_times = []
                for run in range(NUM_RUNS):
                    total_shared = 0
                    for i in range(0, len(typos), batch_size):
                        batch = typos[i:i + batch_size]
                        start = time.perf_counter()
                        _ = checker_shared.find_corrections_batch(batch, algo_key)
                        cp.cuda.Stream.null.synchronize()
                        total_shared += time.perf_counter() - start
                    ms_per_word = total_shared / len(typos) * 1000
                    run_times.append(ms_per_word)

                ms_shared = statistics.median(run_times)

                speedup_shared_vs_global = ms_global / ms_shared if ms_shared > 0 else 0
                speedup_shared_vs_c = ms_c / ms_shared if ms_shared > 0 else 0
                speedup_global_vs_c = ms_c / ms_global if ms_global > 0 else 0

                print(f"    {algo_name}: C={ms_c:.3f}ms, Global={ms_global:.3f}ms ({speedup_global_vs_c:.1f}x), Shared={ms_shared:.3f}ms ({speedup_shared_vs_c:.1f}x) [median of {NUM_RUNS} runs]")

                all_results.append({
                    'algorithm': algo_name, 'language': lang, 'size': size_label, 'num_typos': len(typos),
                    'c_ms': ms_c, 'cuda_global_ms': ms_global, 'cuda_shared_ms': ms_shared,
                    'speedup_global_vs_c': speedup_global_vs_c,
                    'speedup_shared_vs_c': speedup_shared_vs_c,
                    'speedup_shared_vs_global': speedup_shared_vs_global
                })

    # Summary
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Size':<6} | {'Typos':<6} | {'C (ms)':<10} | {'Global (ms)':<12} | {'Shared (ms)':<12} | {'Global/C':<10} | {'Shared/C':<10} | {'Sh/Gl':<8}")
    print("-" * 140)
    for r in all_results:
        print(f"{r['algorithm']:<22} | {r['language']:<4} | {r['size']:<6} | {r['num_typos']:<6} | {r['c_ms']:>8.4f} | {r['cuda_global_ms']:>10.4f} | {r['cuda_shared_ms']:>10.4f} | {r['speedup_global_vs_c']:>8.1f}x | {r['speedup_shared_vs_c']:>8.1f}x | {r['speedup_shared_vs_global']:>6.2f}x")

    # Save results
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    output = {'description': 'CUDA Shared Memory vs Global vs C', 'device': device_name, 'num_runs': NUM_RUNS, 'results': all_results}
    with open(results_dir / "cuda_shared_vs_c.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    # CSV
    with open(results_dir / "cuda_shared_vs_c.csv", 'w', encoding='utf-8') as f:
        f.write("Algorithm,Language,Size,Num_Typos,C_ms,CUDA_Global_ms,CUDA_Shared_ms,Speedup_Global_vs_C,Speedup_Shared_vs_C,Speedup_Shared_vs_Global\n")
        for r in all_results:
            f.write(f"{r['algorithm']},{r['language']},{r['size']},{r['num_typos']},{r['c_ms']:.4f},{r['cuda_global_ms']:.4f},{r['cuda_shared_ms']:.4f},{r['speedup_global_vs_c']:.2f},{r['speedup_shared_vs_c']:.2f},{r['speedup_shared_vs_global']:.2f}\n")

    print(f"\nResults saved to: {results_dir / 'cuda_shared_vs_c.json'}")
    print(f"CSV saved to: {results_dir / 'cuda_shared_vs_c.csv'}")


if __name__ == "__main__":
    main()
