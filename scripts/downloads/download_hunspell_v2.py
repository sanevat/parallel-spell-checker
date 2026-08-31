#!/usr/bin/env python3
"""
Download Hunspell dictionaries v2 - Try alternative sources.

Uses GitHub repos that mirror Hunspell dictionaries.
"""

import os
import sys
import re
import tempfile
import urllib.request
import ssl
from pathlib import Path
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MK_CYRILLIC = set('абвгдѓежзѕијклљмнњопрстќуфхцчџш')
EN_LATIN = set('abcdefghijklmnopqrstuvwxyz')


def download_url(url: str, timeout: int = 60) -> bytes:
    """Download URL and return content."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
        return response.read()


def parse_dic_content(content: str, valid_chars: set) -> list:
    """Parse .dic file content."""
    words = []
    lines = content.strip().split('\n')

    # Skip first line (word count)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # Strip flags after /
        if '/' in line:
            word = line.split('/')[0]
        else:
            word = line

        # Strip tabs
        word = word.split('\t')[0].strip().lower()

        if word and all(c in valid_chars for c in word):
            words.append(word)

    return words


def try_download_mk():
    """Try multiple sources for Macedonian dictionary."""
    print("\n=== Trying Macedonian Dictionary Sources ===")

    sources = [
        # wooorm dictionaries (comprehensive collection)
        ("wooorm/dictionaries", "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/mk/index.dic"),

        # LibreOffice dictionaries (different path formats)
        ("LibreOffice/mk_MK", "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/mk_MK/mk_MK.dic"),

        # Titusz's collection
        ("titusz/hunspell-dict", "https://raw.githubusercontent.com/titusz/python-hunspell-dict/master/dictionaries/mk_MK.dic"),

        # Alternative mirrors
        ("SublimeLinter", "https://raw.githubusercontent.com/SublimeLinter/SublimeLinter-contrib-hunspell/master/dictionaries/mk_MK.dic"),
    ]

    for name, url in sources:
        print(f"\nTrying {name}...")
        print(f"  URL: {url}")
        try:
            data = download_url(url)
            # Try different encodings
            for enc in ['utf-8', 'utf-8-sig', 'cp1251', 'iso-8859-5']:
                try:
                    content = data.decode(enc)
                    words = parse_dic_content(content, MK_CYRILLIC)
                    if words:
                        print(f"  SUCCESS! Decoded with {enc}, got {len(words):,} words")
                        return words
                except UnicodeDecodeError:
                    continue
            print(f"  Downloaded but no valid words found")
        except Exception as e:
            print(f"  ERROR: {e}")

    return []


def try_download_en():
    """Try multiple sources for English dictionary."""
    print("\n=== Trying English Dictionary Sources ===")

    sources = [
        # wooorm dictionaries
        ("wooorm/en", "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/en/index.dic"),

        # SCOWL-based larger dictionary
        ("en-wl/wordlist", "https://raw.githubusercontent.com/en-wl/wordlist/master/alt12dicts/2of12.txt"),

        # LibreOffice
        ("LibreOffice/en_US", "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_US.dic"),

        # Wikimedia word frequency list
        ("other", "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"),
    ]

    all_words = set()

    for name, url in sources:
        print(f"\nTrying {name}...")
        print(f"  URL: {url}")
        try:
            data = download_url(url)
            content = data.decode('utf-8')

            # Different formats
            if 'wooorm' in url or 'LibreOffice' in url:
                words = parse_dic_content(content, EN_LATIN)
            else:
                # Simple word list format
                words = []
                for line in content.strip().split('\n'):
                    word = line.strip().lower()
                    if word and all(c in EN_LATIN for c in word):
                        words.append(word)

            if words:
                print(f"  Got {len(words):,} words")
                all_words.update(words)
        except Exception as e:
            print(f"  ERROR: {e}")

    return sorted(all_words)


def print_stats(words: list, name: str, valid_chars: set):
    """Print dictionary statistics."""
    print(f"\n{'='*60}")
    print(f"{name} Dictionary Statistics")
    print(f"{'='*60}")

    print(f"Total words: {len(words):,}")

    lengths = Counter(len(w) for w in words)
    print("\nWord length distribution:")
    for length in sorted(lengths.keys())[:15]:
        count = lengths[length]
        pct = count / len(words) * 100
        bar = '#' * min(40, int(pct * 2))
        print(f"  {length:2d} chars: {count:6,} ({pct:5.1f}%) {bar}")

    if max(lengths.keys()) > 15:
        print(f"  ... (max length: {max(lengths.keys())})")

    used_chars = set()
    for word in words:
        used_chars.update(word)

    print(f"\nAlphabet: {''.join(sorted(used_chars))}")
    print(f"Coverage: {len(used_chars)}/{len(valid_chars)} expected chars")

    print(f"\nSample words: {', '.join(words[:15])}")


def main():
    print("="*70)
    print("HUNSPELL DICTIONARY DOWNLOADER v2")
    print("="*70)

    base_dir = Path(__file__).parent.parent
    dict_dir = base_dir / "data" / "dictionary"

    # Try Macedonian
    mk_words = try_download_mk()

    # If download failed, use existing filtered
    if not mk_words:
        print("\nMacedonian download failed. Using existing dictionary...")
        existing = dict_dir / "mk_dictionary.txt"
        if existing.exists():
            with open(existing, 'r', encoding='utf-8') as f:
                mk_words = [line.strip().lower() for line in f
                           if all(c in MK_CYRILLIC for c in line.strip().lower())]
            print(f"  Loaded {len(mk_words):,} words from existing")

    # Try English
    en_words = try_download_en()

    # Save results
    if mk_words:
        mk_words = sorted(set(mk_words))
        mk_path = dict_dir / "mk_hunspell.txt"
        with open(mk_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(mk_words))
        print(f"\nSaved MK: {mk_path} ({len(mk_words):,} words)")
        print_stats(mk_words, "Macedonian", MK_CYRILLIC)

    if en_words:
        en_words = sorted(set(en_words))
        en_path = dict_dir / "en_hunspell.txt"
        with open(en_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(en_words))
        print(f"\nSaved EN: {en_path} ({len(en_words):,} words)")
        print_stats(en_words, "English", EN_LATIN)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"MK Hunspell: {len(mk_words):,} words")
    print(f"EN Hunspell: {len(en_words):,} words")


if __name__ == "__main__":
    main()
