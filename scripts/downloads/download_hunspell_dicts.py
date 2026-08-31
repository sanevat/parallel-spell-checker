#!/usr/bin/env python3
"""
Download and parse official Hunspell dictionaries.

Sources:
- Macedonian: OpenOffice contrib dictionaries
- English: LibreOffice dictionaries repository

Hunspell .dic format:
- First line: word count
- Following lines: word[/flags]
- Flags are optional suffixes like /ABC that indicate morphological rules
"""

import os
import sys
import re
import zipfile
import tempfile
import urllib.request
import ssl
from pathlib import Path
from collections import Counter

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Macedonian Cyrillic alphabet (lowercase only)
MK_CYRILLIC = set('абвгдѓежзѕијклљмнњопрстќуфхцчџш')

# English alphabet (lowercase only)
EN_LATIN = set('abcdefghijklmnopqrstuvwxyz')


def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """Download a file from URL."""
    print(f"  Downloading: {url}")

    # Create SSL context that doesn't verify certificates (for some servers)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            data = response.read()
            with open(dest_path, 'wb') as f:
                f.write(data)
        print(f"  Downloaded: {len(data):,} bytes")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def parse_hunspell_dic(filepath: str, valid_chars: set, encoding: str = 'utf-8') -> list:
    """
    Parse a Hunspell .dic file and extract words.

    Format:
    - First line: word count (skip)
    - Following lines: word[/flags]

    Returns list of valid lowercase words.
    """
    words = []

    # Try different encodings
    encodings = [encoding, 'utf-8', 'iso-8859-1', 'cp1251', 'utf-8-sig']

    content = None
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if content is None:
        print(f"  ERROR: Could not decode {filepath}")
        return words

    lines = content.strip().split('\n')

    # Skip first line (word count)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # Strip flags (everything after /)
        if '/' in line:
            word = line.split('/')[0]
        else:
            word = line

        # Strip any trailing whitespace or tab-separated content
        word = word.split('\t')[0].strip()

        # Convert to lowercase
        word = word.lower()

        # Check if all characters are valid
        if word and all(c in valid_chars for c in word):
            words.append(word)

    return words


def download_mk_dictionary(output_path: str) -> list:
    """Download and parse Macedonian Hunspell dictionary."""
    print("\n=== Macedonian Hunspell Dictionary ===")

    # Primary URL
    urls = [
        "http://ftp.osuosl.org/pub/openoffice/contrib/dictionaries/mk_MK.zip",
        "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/mk_MK/mk_MK.dic",
        "https://cgit.freedesktop.org/libreoffice/dictionaries/plain/mk_MK/mk_MK.dic",
    ]

    words = []

    # Try ZIP download first
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "mk_MK.zip")
        dic_path = os.path.join(tmpdir, "mk_MK.dic")

        # Try to download ZIP
        print("\nTrying ZIP download...")
        if download_file(urls[0], zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # List contents
                    print(f"  ZIP contents: {zf.namelist()}")

                    # Find .dic file
                    dic_files = [n for n in zf.namelist() if n.endswith('.dic')]
                    if dic_files:
                        dic_name = dic_files[0]
                        zf.extract(dic_name, tmpdir)
                        extracted_path = os.path.join(tmpdir, dic_name)
                        words = parse_hunspell_dic(extracted_path, MK_CYRILLIC, encoding='utf-8')
            except zipfile.BadZipFile:
                print("  ERROR: Invalid ZIP file")

        # If ZIP failed, try direct .dic download
        if not words:
            print("\nTrying direct .dic download...")
            for url in urls[1:]:
                if download_file(url, dic_path):
                    words = parse_hunspell_dic(dic_path, MK_CYRILLIC, encoding='utf-8')
                    if words:
                        break

    # If downloads failed, try to generate from existing dictionary
    if not words:
        print("\nDirect downloads failed. Trying to filter existing dictionary...")
        existing_path = Path(output_path).parent / "mk_dictionary.txt"
        if existing_path.exists():
            with open(existing_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and all(c in MK_CYRILLIC for c in word):
                        words.append(word)
            print(f"  Filtered {len(words):,} words from existing dictionary")

    # Remove duplicates and sort
    words = sorted(set(words))

    # Save
    if words:
        with open(output_path, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word + '\n')
        print(f"\nSaved: {output_path}")
        print(f"Total words: {len(words):,}")

    return words


def download_en_dictionary(output_path: str) -> list:
    """Download and parse English Hunspell dictionary."""
    print("\n=== English Hunspell Dictionary ===")

    urls = [
        "https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.dic",
        "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_US.dic",
        "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/en/index.dic",
    ]

    words = []

    with tempfile.TemporaryDirectory() as tmpdir:
        dic_path = os.path.join(tmpdir, "en_US.dic")

        for url in urls:
            print(f"\nTrying: {url}")
            if download_file(url, dic_path):
                words = parse_hunspell_dic(dic_path, EN_LATIN, encoding='utf-8')
                if words:
                    break

    # If downloads failed, try to filter existing dictionary
    if not words:
        print("\nDirect downloads failed. Trying to filter existing dictionary...")
        existing_path = Path(output_path).parent / "en_dictionary.txt"
        if existing_path.exists():
            with open(existing_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and all(c in EN_LATIN for c in word):
                        words.append(word)
            print(f"  Filtered {len(words):,} words from existing dictionary")

    # Remove duplicates and sort
    words = sorted(set(words))

    # Save
    if words:
        with open(output_path, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word + '\n')
        print(f"\nSaved: {output_path}")
        print(f"Total words: {len(words):,}")

    return words


def print_statistics(words: list, name: str, valid_chars: set):
    """Print statistics about a dictionary."""
    print(f"\n{'='*60}")
    print(f"Statistics for {name}")
    print(f"{'='*60}")

    print(f"\nTotal words: {len(words):,}")

    # Word length distribution
    lengths = Counter(len(w) for w in words)
    print("\nWord length distribution:")
    for length in sorted(lengths.keys()):
        count = lengths[length]
        pct = count / len(words) * 100
        bar = '#' * min(50, int(pct * 2))
        print(f"  {length:2d} chars: {count:6,} ({pct:5.1f}%) {bar}")

    # Statistics
    avg_len = sum(len(w) for w in words) / len(words) if words else 0
    min_len = min(len(w) for w in words) if words else 0
    max_len = max(len(w) for w in words) if words else 0
    print(f"\nLength stats: min={min_len}, max={max_len}, avg={avg_len:.1f}")

    # Alphabet coverage
    used_chars = set()
    for word in words:
        used_chars.update(word)

    print(f"\nAlphabet coverage:")
    print(f"  Expected: {len(valid_chars)} characters")
    print(f"  Used:     {len(used_chars)} characters")
    print(f"  Characters: {''.join(sorted(used_chars))}")

    missing = valid_chars - used_chars
    if missing:
        print(f"  Missing:  {''.join(sorted(missing))}")

    extra = used_chars - valid_chars
    if extra:
        print(f"  Extra:    {''.join(sorted(extra))}")

    # Sample words
    print(f"\nSample words (first 10):")
    for word in words[:10]:
        print(f"  {word}")

    print(f"\nSample words (random 10 from middle):")
    mid = len(words) // 2
    for word in words[mid:mid+10]:
        print(f"  {word}")


def main():
    print("="*70)
    print("HUNSPELL DICTIONARY DOWNLOADER")
    print("="*70)

    base_dir = Path(__file__).parent.parent
    dict_dir = base_dir / "data" / "dictionary"
    dict_dir.mkdir(parents=True, exist_ok=True)

    mk_output = dict_dir / "mk_hunspell.txt"
    en_output = dict_dir / "en_hunspell.txt"

    # Download Macedonian
    mk_words = download_mk_dictionary(str(mk_output))

    # Download English
    en_words = download_en_dictionary(str(en_output))

    # Print statistics
    if mk_words:
        print_statistics(mk_words, "Macedonian Hunspell", MK_CYRILLIC)

    if en_words:
        print_statistics(en_words, "English Hunspell", EN_LATIN)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"MK Hunspell dictionary: {len(mk_words):,} words -> {mk_output}")
    print(f"EN Hunspell dictionary: {len(en_words):,} words -> {en_output}")
    print("="*70)


if __name__ == "__main__":
    main()
