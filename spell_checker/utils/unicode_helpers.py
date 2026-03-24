"""
UTF-8 / Unicode handling utilities for spell checker.

These utilities ensure algorithms work with Unicode code points,
not raw bytes, which is essential for Cyrillic and other non-ASCII text.
"""

from typing import List, Tuple


def to_codepoints(text: str) -> List[int]:
    """
    Convert a string to a list of Unicode code points.

    This ensures we work with actual characters, not UTF-8 bytes.
    For example, Cyrillic 'а' (U+0430) becomes [1072], not [208, 176].

    Args:
        text: Input string (any Unicode)

    Returns:
        List of integer code points
    """
    return [ord(c) for c in text]


def from_codepoints(codepoints: List[int]) -> str:
    """
    Convert a list of Unicode code points back to a string.

    Args:
        codepoints: List of integer code points

    Returns:
        Reconstructed string
    """
    return ''.join(chr(cp) for cp in codepoints)


def is_cyrillic(char: str) -> bool:
    """
    Check if a character is in the Cyrillic Unicode block.

    Cyrillic range: U+0400 to U+04FF (basic Cyrillic)
    Extended Cyrillic: U+0500 to U+052F

    Args:
        char: Single character

    Returns:
        True if character is Cyrillic
    """
    if len(char) != 1:
        return False
    cp = ord(char)
    return (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F)


def is_ascii_alpha(char: str) -> bool:
    """
    Check if a character is an ASCII letter (a-z, A-Z).

    Args:
        char: Single character

    Returns:
        True if character is ASCII letter
    """
    if len(char) != 1:
        return False
    cp = ord(char)
    return (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A)


def get_script(text: str) -> str:
    """
    Detect the primary script of a text string.

    Args:
        text: Input string

    Returns:
        'cyrillic', 'latin', or 'mixed'
    """
    cyrillic_count = sum(1 for c in text if is_cyrillic(c))
    ascii_count = sum(1 for c in text if is_ascii_alpha(c))

    if cyrillic_count > 0 and ascii_count == 0:
        return 'cyrillic'
    elif ascii_count > 0 and cyrillic_count == 0:
        return 'latin'
    elif cyrillic_count > 0 or ascii_count > 0:
        return 'mixed'
    return 'unknown'


def normalize_text(text: str) -> str:
    """
    Normalize text for spell checking.

    - Converts to lowercase
    - Strips whitespace
    - Applies Unicode NFC normalization

    Args:
        text: Input string

    Returns:
        Normalized string
    """
    import unicodedata
    return unicodedata.normalize('NFC', text.lower().strip())


def char_info(char: str) -> dict:
    """
    Get detailed information about a character (useful for debugging).

    Args:
        char: Single character

    Returns:
        Dictionary with character info
    """
    import unicodedata
    if len(char) != 1:
        raise ValueError("Expected single character")

    return {
        'char': char,
        'codepoint': ord(char),
        'hex': f'U+{ord(char):04X}',
        'name': unicodedata.name(char, 'UNKNOWN'),
        'category': unicodedata.category(char),
        'is_cyrillic': is_cyrillic(char),
        'is_ascii': is_ascii_alpha(char),
        'utf8_bytes': len(char.encode('utf-8'))
    }
