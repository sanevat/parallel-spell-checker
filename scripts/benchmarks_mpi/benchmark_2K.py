#!/usr/bin/env python3
"""
Generate test files with exactly 2000 misspelled words and run sequential benchmark.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from collections import Counter, defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector

# Alphabets
MK_ALPHABET = 'абвгдѓежзѕијклљмнњопрстќуфхцчџш'
EN_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

# Error type distribution
ERROR_WEIGHTS = {'swap': 17.9, 'replace': 30.8, 'insert': 35.1, 'delete': 16.2}

random.seed(42)


def load_dictionary(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def introduce_typo(word: str, alphabet: str) -> tuple:
    if len(word) < 3:
        return None, None

    error_types = list(ERROR_WEIGHTS.keys())
    weights = list(ERROR_WEIGHTS.values())
    error_type = random.choices(error_types, weights=weights, k=1)[0]

    chars = list(word)

    if error_type == 'swap':
        if len(chars) < 2:
            return None, None
        pos = random.randint(0, len(chars) - 2)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        if chars[pos] == chars[pos + 1]:
            return None, None

    elif error_type == 'replace':
        pos = random.randint(0, len(chars) - 1)
        old_char = chars[pos]
        new_char = random.choice([c for c in alphabet if c != old_char])
        chars[pos] = new_char

    elif error_type == 'insert':
        pos = random.randint(0, len(chars))
        new_char = random.choice(alphabet)
        chars.insert(pos, new_char)

    elif error_type == 'delete':
        if len(chars) < 3:
            return None, None
        pos = random.randint(0, len(chars) - 1)
        del chars[pos]

    typo = ''.join(chars)
    if typo == word:
        return None, None

    return typo, error_type


def generate_test_data(dictionary: list, alphabet: str, num_typos: int = 2000):
    """Generate test data with exactly num_typos misspelled words."""
    dict_set = set(dictionary)

    # Target: 35% typos, 65% correct
    # If 2000 typos = 35%, then total = 2000/0.35 = 5714
    num_correct = int(num_typos * 65 / 35)
    total_words = num_typos + num_correct

    print(f"  Target: {num_typos} typos + {num_correct} correct = {total_words} total")

    words = []
    ground_truth = {}
    error_counts = Counter()

    # Generate typos first
    typo_count = 0
    attempts = 0
    max_attempts = num_typos * 10

    while typo_count < num_typos and attempts < max_attempts:
        word = random.choice(dictionary)
        if len(word) < 3:
            attempts += 1
            continue

        typo, error_type = introduce_typo(word, alphabet)

        if typo and typo not in dict_set and typo not in ground_truth:
            words.append(typo)
            ground_truth[typo] = word
            error_counts[error_type] += 1
            typo_count += 1

        attempts += 1

    print(f"  Generated {typo_count} unique typos")

    # Add correct words
    correct_words = random.choices(
        [w for w in dictionary if len(w) >= 3],
        k=num_correct
    )
    words.extend(correct_words)

    # Shuffle
    random.shuffle(words)

    text = ' '.join(words)

    # Count actual stats
    actual_typos = sum(1 for w in words if w in ground_truth)
    actual_correct = len(words) - actual_typos

    stats = {
        'total_words': len(words),
        'typo_words': actual_typos,
        'correct_words': actual_correct,
        'unique_typos': len(ground_truth),
        'error_types': dict(error_counts),
    }

    return text, ground_truth, stats


def verify_data(text: str, ground_truth: dict, dict_set: set, name: str) -> bool:
    print(f"\n  Verifying {name}...")
    words = text.split()

    typos_not_in_dict = 0
    correct_in_dict = 0
    errors = []

    for word in words:
        if word in ground_truth:
            if word not in dict_set:
                typos_not_in_dict += 1
            else:
                errors.append(f"Typo '{word}' in dict!")
        else:
            if word in dict_set:
                correct_in_dict += 1
            else:
                if len(errors) < 3:
                    errors.append(f"Correct '{word}' not in dict!")

    gt_valid = sum(1 for c in ground_truth.values() if c in dict_set)

    print(f"    Typos not in dict: {typos_not_in_dict}")
    print(f"    Correct in dict: {correct_in_dict}")
    print(f"    GT corrections valid: {gt_valid}/{len(ground_truth)}")

    if errors:
        print(f"    ERRORS: {errors}")
        return False
    print(f"    PASSED!")
    return True


def group_by_length(words: list) -> dict:
    by_len = defaultdict(list)
    for w in words:
        by_len[len(w)].append(w)
    return dict(by_len)


def get_candidates(word_len: int, by_len: dict, tol: int = 2) -> list:
    candidates = []
    for length in range(max(1, word_len - tol), word_len + tol + 1):
        if length in by_len:
            candidates.extend(by_len[length])
    return candidates


def find_best_match(typo: str, candidates: list, dist_fn) -> tuple:
    if not candidates:
        return "", 999999
    best = candidates[0]
    best_d = dist_fn(typo, candidates[0])
    for c in candidates[1:]:
        d = dist_fn(typo, c)
        if d < best_d:
            best_d = d
            best = c
            if d == 0:
                break
    return best, best_d


def run_benchmark(typos: list, dict_by_len: dict, ground_truth: dict,
                  algo_name: str, algo_fn, lang: str) -> dict:
    print(f"\n{'='*70}")
    print(f"{algo_name} + {lang} ({len(typos)} misspelled words)")
    print(f"{'='*70}")

    total_time = 0.0
    total_candidates = 0
    correct = 0

    start_all = time.perf_counter()

    for i, typo in enumerate(typos):
        candidates = get_candidates(len(typo), dict_by_len)
        total_candidates += len(candidates)

        start = time.perf_counter()
        best, _ = find_best_match(typo, candidates, algo_fn)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if typo in ground_truth and best == ground_truth[typo]:
            correct += 1

        # Progress every 100 words
        if (i + 1) % 100 == 0 or i == 0:
            avg_ms = total_time / (i + 1) * 1000
            remaining = len(typos) - (i + 1)
            eta_s = remaining * avg_ms / 1000
            eta_m = int(eta_s // 60)
            eta_sec = int(eta_s % 60)
            pct = (i + 1) / len(typos) * 100
            print(f"  {algo_name[:3]}+{lang}: {i+1}/{len(typos)} ({pct:.0f}%) "
                  f"avg {avg_ms:.1f}ms/word ETA {eta_m}m {eta_sec}s")

    total_elapsed = time.perf_counter() - start_all
    accuracy = correct / len(typos) * 100
    ms_per_word = total_time / len(typos) * 1000
    avg_candidates = total_candidates / len(typos)

    print(f"\n  DONE: {total_time:.1f}s total, {ms_per_word:.1f}ms/word, "
          f"{accuracy:.1f}% accuracy ({correct}/{len(typos)})")

    return {
        'algorithm': algo_name,
        'language': lang,
        'misspelled': len(typos),
        'total_time_s': total_time,
        'ms_per_word': ms_per_word,
        'correct': correct,
        'accuracy_pct': accuracy,
        'avg_candidates': avg_candidates,
    }


def main():
    print("=" * 70)
    print("SEQUENTIAL BENCHMARK - 2000 MISSPELLED WORDS")
    print("=" * 70)

    base = Path(__file__).parent.parent

    # Load dictionaries
    print("\nLoading dictionaries...")
    mk_dict = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
    en_dict = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
    print(f"  MK: {len(mk_dict):,} words")
    print(f"  EN: {len(en_dict):,} words")

    mk_dict_set = set(mk_dict)
    en_dict_set = set(en_dict)

    # Generate test data
    print("\n" + "=" * 70)
    print("GENERATING TEST DATA (2000 typos each)")
    print("=" * 70)

    print("\nGenerating MK data...")
    mk_text, mk_gt, mk_stats = generate_test_data(mk_dict, MK_ALPHABET, 2000)
    print(f"  Stats: {mk_stats}")

    print("\nGenerating EN data...")
    en_text, en_gt, en_stats = generate_test_data(en_dict, EN_ALPHABET, 2000)
    print(f"  Stats: {en_stats}")

    # Verify
    print("\n" + "=" * 70)
    print("VERIFYING DATA")
    print("=" * 70)

    mk_ok = verify_data(mk_text, mk_gt, mk_dict_set, "MK")
    en_ok = verify_data(en_text, en_gt, en_dict_set, "EN")

    if not (mk_ok and en_ok):
        print("VERIFICATION FAILED!")
        return

    # Save test files
    print("\n" + "=" * 70)
    print("SAVING TEST FILES")
    print("=" * 70)

    mk_text_path = base / "data/test_texts/macedonian/mk_hunspell_typos_2K.txt"
    mk_gt_path = base / "data/ground_truth/mk_hunspell_corrections_2K.json"
    en_text_path = base / "data/test_texts/english/en_hunspell_typos_2K.txt"
    en_gt_path = base / "data/ground_truth/en_hunspell_corrections_2K.json"

    with open(mk_text_path, 'w', encoding='utf-8') as f:
        f.write(mk_text)
    print(f"  Saved: {mk_text_path.name}")

    with open(mk_gt_path, 'w', encoding='utf-8') as f:
        json.dump(mk_gt, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {mk_gt_path.name}")

    with open(en_text_path, 'w', encoding='utf-8') as f:
        f.write(en_text)
    print(f"  Saved: {en_text_path.name}")

    with open(en_gt_path, 'w', encoding='utf-8') as f:
        json.dump(en_gt, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {en_gt_path.name}")

    print(f"\nMK: {mk_stats['total_words']} total, {mk_stats['typo_words']} misspelled, {mk_stats['correct_words']} correct")
    print(f"EN: {en_stats['total_words']} total, {en_stats['typo_words']} misspelled, {en_stats['correct_words']} correct")

    # Prepare for benchmark
    print("\n" + "=" * 70)
    print("PREPARING BENCHMARK")
    print("=" * 70)

    mk_by_len = group_by_length(mk_dict)
    en_by_len = group_by_length(en_dict)

    # Get typos (words not in dictionary)
    mk_words = mk_text.split()
    en_words = en_text.split()

    mk_typos = [w for w in mk_words if w not in mk_dict_set]
    en_typos = [w for w in en_words if w not in en_dict_set]

    print(f"  MK typos to process: {len(mk_typos)}")
    print(f"  EN typos to process: {len(en_typos)}")

    # Sample candidate counts
    sample_mk = mk_typos[0] if mk_typos else ""
    sample_en = en_typos[0] if en_typos else ""
    print(f"  Sample MK '{sample_mk}' ({len(sample_mk)} chars): {len(get_candidates(len(sample_mk), mk_by_len)):,} candidates")
    print(f"  Sample EN '{sample_en}' ({len(sample_en)} chars): {len(get_candidates(len(sample_en), en_by_len)):,} candidates")

    # Run benchmarks_mpi
    algorithms = [
        ("Levenshtein", levenshtein),
        ("Damerau-Levenshtein", damerau_levenshtein),
        ("Myers Bit-Vector", myers_bitvector),
    ]

    results = []

    for algo_name, algo_fn in algorithms:
        # MK
        r = run_benchmark(mk_typos, mk_by_len, mk_gt, algo_name, algo_fn, "MK")
        results.append(r)

        # EN
        r = run_benchmark(en_typos, en_by_len, en_gt, algo_name, algo_fn, "EN")
        results.append(r)

    # Print results table
    print("\n" + "=" * 100)
    print("SEQUENTIAL BASELINE (2000 misspelled words, 50K dict)")
    print("=" * 100)
    print(f"{'Algorithm':<22} | {'Lang':<4} | {'Misspelled':>10} | {'Time(s)':>8} | {'ms/word':>8} | {'Accuracy':>10}")
    print("-" * 100)

    for r in results:
        acc_str = f"{r['accuracy_pct']:.1f}% ({r['correct']}/{r['misspelled']})"
        print(f"{r['algorithm']:<22} | {r['language']:<4} | {r['misspelled']:>10,} | "
              f"{r['total_time_s']:>8.1f} | {r['ms_per_word']:>8.1f} | {acc_str:>10}")

    print("-" * 100)

    # Save results
    results_file = base / "results" / "sequential_2K.json"
    results_file.parent.mkdir(exist_ok=True)

    output = {
        'description': 'Sequential baseline with 2000 misspelled words, 50K Hunspell dict',
        'mk_stats': mk_stats,
        'en_stats': en_stats,
        'results': results,
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
