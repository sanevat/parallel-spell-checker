#!/usr/bin/env python3
"""
Generate 250-word test files with typos from dictionary.
Creates exactly 250 typos like the 500/1K/2K versions.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent

# Error type distributions (same as other benchmarks_mpi)
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


def generate_test_files(num_typos: int = 250):
    """Generate test files with exactly num_typos typos."""
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

        # Add correct words (~1.857 ratio like other benchmarks_mpi)
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
        typo_file = BASE / f"data/test_texts/{'macedonian' if lang == 'mk' else 'english'}/{lang}_hunspell_typos_250.txt"
        gt_file = BASE / f"data/ground_truth/{lang}_hunspell_corrections_250.json"

        with open(typo_file, 'w', encoding='utf-8') as f:
            f.write(text)

        with open(gt_file, 'w', encoding='utf-8') as f:
            json.dump(corrections, f, ensure_ascii=False, indent=2)

        print(f"  Typos: {len(typos)}, Correct: {len(correct_words)}, Total: {len(all_words)}")
        print(f"  Error types: {dict(error_counts)}")
        print(f"  Saved: {typo_file.name}, {gt_file.name}")


def main():
    random.seed(42)
    generate_test_files(250)
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
