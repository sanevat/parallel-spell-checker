#!/usr/bin/env python3
"""
Generate Turkish test files with typos from dictionary.
Creates files for all sizes: 250, 500, 1K, 1500, 2K, 2500, 3500, 5000

Error types distribution:
- Insert random char: 35.1%
- Replace with random char: 30.8%
- Swap adjacent characters: 17.9%
- Delete random char: 16.2%
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

BASE = Path(__file__).parent.parent.parent

ERROR_TYPES = {
    'insert': 0.351,
    'replace': 0.308,
    'swap': 0.179,
    'delete': 0.162
}

# Turkish alphabet (lowercase)
TR_CHARS = 'abcçdefgğhıijklmnoöprsştuüvyz'


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


def generate_test_files(num_typos: int, dict_tr: List[str], dict_tr_set: set, tr_by_len: Dict):
    print(f"\nGenerating {num_typos}-word Turkish test file...")

    tr_valid = [w for w in dict_tr if 4 <= len(w) <= 12]
    random.shuffle(tr_valid)

    typos = []
    corrections = {}
    error_counts = defaultdict(int)
    attempts = 0
    max_attempts = num_typos * 50

    for word in tr_valid:
        if len(typos) >= num_typos:
            break
        if attempts > max_attempts:
            break
        attempts += 1

        typo, error_type = generate_typo(word, TR_CHARS)

        if typo not in dict_tr_set and word in dict_tr_set and typo != word:
            candidates = get_candidates(len(typo), tr_by_len)
            if word in candidates:
                typos.append(typo)
                corrections[typo] = word
                error_counts[error_type] += 1

    num_correct = int(num_typos * 1.857)
    correct_words = [w for w in tr_valid if w in dict_tr_set and w not in corrections.values()][:num_correct]

    all_words = typos + correct_words
    random.shuffle(all_words)

    text_lines = []
    words_per_line = 10
    for i in range(0, len(all_words), words_per_line):
        line = ' '.join(all_words[i:i+words_per_line])
        text_lines.append(line + '.')

    text = '\n'.join(text_lines)

    # Create directory if needed
    tr_dir = BASE / "data/test_texts/turkish"
    tr_dir.mkdir(parents=True, exist_ok=True)

    # Determine file suffix
    if num_typos >= 1000:
        suffix = f"{num_typos // 1000}K" if num_typos % 1000 == 0 else str(num_typos)
    else:
        suffix = str(num_typos)

    typo_file = tr_dir / f"tr_hunspell_typos_{suffix}.txt"
    gt_file = BASE / f"data/ground_truth/tr_hunspell_corrections_{suffix}.json"

    with open(typo_file, 'w', encoding='utf-8') as f:
        f.write(text)

    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)

    print(f"  Typos: {len(typos)}, Correct: {len(correct_words)}, Total: {len(all_words)}")
    print(f"  Error types: {dict(error_counts)}")
    print(f"  Saved: {typo_file.name}, {gt_file.name}")


def main():
    random.seed(42)

    print("=" * 80)
    print("GENERATING TURKISH TEST FILES")
    print("=" * 80)

    # Load dictionary once
    dict_tr = load_dictionary(str(BASE / "data/dictionary/tr_equal.txt"))
    print(f"Loaded TR: {len(dict_tr):,} words")

    dict_tr_set = set(dict_tr)
    tr_by_len = group_by_length(dict_tr)

    # Generate all sizes matching MK/EN
    sizes = [250, 500, 1000, 1500, 2000, 2500, 3500, 5000]

    for size in sizes:
        generate_test_files(size, dict_tr, dict_tr_set, tr_by_len)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
