#!/usr/bin/env python3
"""
Generate test data specifically for Hunspell equal dictionaries (50K each).

Creates typo files where:
- All correct words exist in the dictionary
- All typos have known corrections in the dictionary
- Ground truth is 100% accurate

Error types distribution:
- Swap adjacent characters: 17.9%
- Replace with random char: 30.8%
- Insert random char: 35.1%
- Delete random char: 16.2%
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Alphabets
MK_ALPHABET = 'абвгдѓежзѕијклљмнњопрстќуфхцчџш'
EN_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

# Error type distribution
ERROR_WEIGHTS = {
    'swap': 17.9,
    'replace': 30.8,
    'insert': 35.1,
    'delete': 16.2,
}

# Seed for reproducibility
random.seed(42)


def load_dictionary(path: str) -> list:
    """Load dictionary words."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def introduce_typo(word: str, alphabet: str) -> tuple:
    """
    Introduce a typo into a word.
    Returns (typo_word, error_type) or (None, None) if can't create typo.
    """
    if len(word) < 2:
        return None, None

    # Choose error type based on weights
    error_types = list(ERROR_WEIGHTS.keys())
    weights = list(ERROR_WEIGHTS.values())
    error_type = random.choices(error_types, weights=weights, k=1)[0]

    chars = list(word)

    if error_type == 'swap':
        # Swap two adjacent characters
        if len(chars) < 2:
            return None, None
        pos = random.randint(0, len(chars) - 2)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        # Make sure swap actually changed something
        if chars[pos] == chars[pos + 1]:
            return None, None

    elif error_type == 'replace':
        # Replace a random character with another from alphabet
        pos = random.randint(0, len(chars) - 1)
        old_char = chars[pos]
        # Pick a different character
        new_char = random.choice([c for c in alphabet if c != old_char])
        chars[pos] = new_char

    elif error_type == 'insert':
        # Insert a random character at random position
        pos = random.randint(0, len(chars))
        new_char = random.choice(alphabet)
        chars.insert(pos, new_char)

    elif error_type == 'delete':
        # Delete a random character
        if len(chars) < 3:  # Keep at least 2 chars
            return None, None
        pos = random.randint(0, len(chars) - 1)
        del chars[pos]

    typo = ''.join(chars)

    # Make sure typo is different from original
    if typo == word:
        return None, None

    return typo, error_type


def generate_test_data(
    dictionary: list,
    alphabet: str,
    target_size_bytes: int,
    typo_ratio: float = 0.35
) -> tuple:
    """
    Generate test data with typos.

    Returns:
        (text, ground_truth, stats)
    """
    dict_set = set(dictionary)

    # Estimate words needed for target size
    avg_word_len = sum(len(w) for w in dictionary) / len(dictionary)
    estimated_words = int(target_size_bytes / (avg_word_len + 1))  # +1 for space

    print(f"  Target size: {target_size_bytes:,} bytes")
    print(f"  Avg word length: {avg_word_len:.1f}")
    print(f"  Estimated words needed: {estimated_words:,}")

    words = []
    ground_truth = {}
    error_type_counts = Counter()
    correct_count = 0
    typo_count = 0

    # Sample words with replacement
    sampled_words = random.choices(dictionary, k=estimated_words)

    for original_word in sampled_words:
        if len(original_word) < 3:
            # Keep short words as-is
            words.append(original_word)
            correct_count += 1
            continue

        # Decide if this word should have a typo
        if random.random() < typo_ratio:
            # Try to create a typo
            typo, error_type = introduce_typo(original_word, alphabet)

            if typo and typo not in dict_set:
                # Valid typo - not in dictionary
                words.append(typo)
                ground_truth[typo] = original_word
                error_type_counts[error_type] += 1
                typo_count += 1
            else:
                # Couldn't create valid typo, keep original
                words.append(original_word)
                correct_count += 1
        else:
            # Keep as correct word
            words.append(original_word)
            correct_count += 1

    # Join words with spaces
    text = ' '.join(words)

    # Adjust if we're far from target size
    actual_size = len(text.encode('utf-8'))
    if actual_size < target_size_bytes * 0.9:
        # Need more words - recursive call or just pad
        print(f"  Size {actual_size:,} < target, adjusting...")

    stats = {
        'total_words': len(words),
        'correct_words': correct_count,
        'typo_words': typo_count,
        'typo_ratio': typo_count / len(words) * 100,
        'text_size_bytes': len(text.encode('utf-8')),
        'error_types': dict(error_type_counts),
        'unique_typos': len(ground_truth),
    }

    return text, ground_truth, stats


def verify_data(
    text: str,
    ground_truth: dict,
    dictionary: list,
    lang: str
) -> bool:
    """Verify generated data is consistent."""
    print(f"\n  Verifying {lang} data...")

    dict_set = set(dictionary)
    words = text.split()

    errors = []

    # Check all words
    correct_in_dict = 0
    typos_not_in_dict = 0
    unknown = 0

    for word in words:
        if word in ground_truth:
            # This is a typo
            if word in dict_set:
                errors.append(f"Typo '{word}' found in dictionary!")
            else:
                typos_not_in_dict += 1
        else:
            # This should be a correct word
            if word in dict_set:
                correct_in_dict += 1
            else:
                unknown += 1
                if len(errors) < 10:
                    errors.append(f"Correct word '{word}' not in dictionary")

    # Check ground truth corrections
    gt_corrections_valid = 0
    for typo, correction in ground_truth.items():
        if correction in dict_set:
            gt_corrections_valid += 1
        else:
            if len(errors) < 10:
                errors.append(f"GT correction '{correction}' not in dictionary")

    print(f"    Correct words in dict: {correct_in_dict:,}")
    print(f"    Typos not in dict: {typos_not_in_dict:,}")
    print(f"    Unknown words: {unknown}")
    print(f"    GT corrections valid: {gt_corrections_valid}/{len(ground_truth)}")

    if errors:
        print(f"    ERRORS ({len(errors)}):")
        for e in errors[:5]:
            print(f"      - {e}")
        return False

    print(f"    VERIFICATION PASSED!")
    return True


def main():
    print("=" * 70)
    print("GENERATE HUNSPELL-COMPATIBLE TEST DATA")
    print("=" * 70)

    base = Path(__file__).parent.parent

    # Load dictionaries
    print("\nLoading dictionaries...")
    mk_dict = load_dictionary(str(base / "data/dictionary/mk_equal.txt"))
    en_dict = load_dictionary(str(base / "data/dictionary/en_equal.txt"))
    print(f"  MK dictionary: {len(mk_dict):,} words")
    print(f"  EN dictionary: {len(en_dict):,} words")

    target_size = 1 * 1024 * 1024  # 1MB

    # Generate Macedonian data
    print("\n" + "=" * 70)
    print("Generating Macedonian test data...")
    print("=" * 70)

    mk_text, mk_gt, mk_stats = generate_test_data(
        mk_dict, MK_ALPHABET, target_size, typo_ratio=0.35
    )

    print(f"\n  MK Statistics:")
    print(f"    Total words: {mk_stats['total_words']:,}")
    print(f"    Correct words: {mk_stats['correct_words']:,} ({100-mk_stats['typo_ratio']:.1f}%)")
    print(f"    Typo words: {mk_stats['typo_words']:,} ({mk_stats['typo_ratio']:.1f}%)")
    print(f"    Unique typos: {mk_stats['unique_typos']:,}")
    print(f"    Text size: {mk_stats['text_size_bytes']:,} bytes")
    print(f"    Error types: {mk_stats['error_types']}")

    # Generate English data
    print("\n" + "=" * 70)
    print("Generating English test data...")
    print("=" * 70)

    en_text, en_gt, en_stats = generate_test_data(
        en_dict, EN_ALPHABET, target_size, typo_ratio=0.35
    )

    print(f"\n  EN Statistics:")
    print(f"    Total words: {en_stats['total_words']:,}")
    print(f"    Correct words: {en_stats['correct_words']:,} ({100-en_stats['typo_ratio']:.1f}%)")
    print(f"    Typo words: {en_stats['typo_words']:,} ({en_stats['typo_ratio']:.1f}%)")
    print(f"    Unique typos: {en_stats['unique_typos']:,}")
    print(f"    Text size: {en_stats['text_size_bytes']:,} bytes")
    print(f"    Error types: {en_stats['error_types']}")

    # Verify data
    print("\n" + "=" * 70)
    print("Verifying generated data...")
    print("=" * 70)

    mk_valid = verify_data(mk_text, mk_gt, mk_dict, "MK")
    en_valid = verify_data(en_text, en_gt, en_dict, "EN")

    if not (mk_valid and en_valid):
        print("\nERROR: Verification failed!")
        return

    # Save files
    print("\n" + "=" * 70)
    print("Saving files...")
    print("=" * 70)

    # MK files
    mk_text_path = base / "data/test_texts/macedonian/mk_hunspell_typos_1MB.txt"
    mk_gt_path = base / "data/ground_truth/mk_hunspell_corrections_1MB.json"

    mk_text_path.parent.mkdir(parents=True, exist_ok=True)
    mk_gt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mk_text_path, 'w', encoding='utf-8') as f:
        f.write(mk_text)
    print(f"  Saved: {mk_text_path}")

    with open(mk_gt_path, 'w', encoding='utf-8') as f:
        json.dump(mk_gt, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {mk_gt_path}")

    # EN files
    en_text_path = base / "data/test_texts/english/en_hunspell_typos_1MB.txt"
    en_gt_path = base / "data/ground_truth/en_hunspell_corrections_1MB.json"

    en_text_path.parent.mkdir(parents=True, exist_ok=True)
    en_gt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(en_text_path, 'w', encoding='utf-8') as f:
        f.write(en_text)
    print(f"  Saved: {en_text_path}")

    with open(en_gt_path, 'w', encoding='utf-8') as f:
        json.dump(en_gt, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {en_gt_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nMK 1MB: {mk_stats['total_words']:,} total words, "
          f"{mk_stats['typo_words']:,} misspelled ({mk_stats['typo_ratio']:.1f}%), "
          f"{mk_stats['correct_words']:,} correct ({100-mk_stats['typo_ratio']:.1f}%)")
    print(f"EN 1MB: {en_stats['total_words']:,} total words, "
          f"{en_stats['typo_words']:,} misspelled ({en_stats['typo_ratio']:.1f}%), "
          f"{en_stats['correct_words']:,} correct ({100-en_stats['typo_ratio']:.1f}%)")

    print("\n" + "-" * 70)
    print("All misspelled words are guaranteed to have corrections in dictionary")
    print("All correct words are guaranteed to exist in dictionary")
    print("-" * 70)


if __name__ == "__main__":
    main()
