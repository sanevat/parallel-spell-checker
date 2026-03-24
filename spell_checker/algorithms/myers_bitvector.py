"""
Myers' Bit-Vector Algorithm for Edit Distance

A bit-parallel algorithm that computes edit distance using bitwise operations.
Originally designed for fixed alphabets, this implementation adapts it for
Unicode by computing pattern masks dynamically.

The algorithm achieves O(n * ceil(m/w)) time complexity where w is the word size
(typically 64 bits). For short patterns (m <= 64), this is effectively O(n).

Note: Myers' algorithm computes the same distance as Levenshtein (no transposition).
The advantage is speed through bit-parallelism.

Works with Unicode code points for proper Cyrillic/UTF-8 support.
"""

from typing import Dict, List, Tuple, Set
import sys


def myers_bitvector(word1: str, word2: str) -> int:
    """
    Compute edit distance using Myers' bit-vector algorithm.

    This is a bit-parallel algorithm that uses bitwise operations to compute
    multiple cells of the DP matrix simultaneously. For patterns up to 64
    characters, it processes one column of the DP matrix per character of
    the text.

    Note: This computes standard Levenshtein distance (insert, delete, substitute).
    It does NOT include transposition - for that use Damerau-Levenshtein.

    Args:
        word1: Pattern string (shorter string is better for performance)
        word2: Text string

    Returns:
        Integer edit distance

    Examples:
        >>> myers_bitvector("kitten", "sitting")
        3
        >>> myers_bitvector("празнк", "празник")  # Missing 'и'
        1
        >>> myers_bitvector("мчака", "мачка")  # No transposition = 2
        2
    """
    # Work with Unicode code points
    pattern = list(word1)
    text = list(word2)

    m = len(pattern)
    n = len(text)

    # Edge cases
    if m == 0:
        return n
    if n == 0:
        return m

    # For very short strings or when pattern is longer than word size,
    # fall back to standard DP (or use block-based approach)
    word_size = 64

    if m > word_size:
        # Use block-based Myers for long patterns
        return _myers_block(pattern, text, word_size)

    # Build pattern masks (Peq): for each character, which positions match
    # Peq[c] has bit i set if pattern[i] == c
    peq: Dict[str, int] = {}

    for i, c in enumerate(pattern):
        if c not in peq:
            peq[c] = 0
        peq[c] |= (1 << i)

    # Initialize bit vectors
    # VP (Vertical Positive): positions where d[i] - d[i-1] = +1
    # VN (Vertical Negative): positions where d[i] - d[i-1] = -1
    vp = (1 << m) - 1  # All 1s: initially all +1 differences
    vn = 0             # All 0s: no -1 differences

    score = m  # Initial distance is length of pattern

    # Mask for the m-th bit
    last_bit = 1 << (m - 1)

    # Process each character of the text
    for c in text:
        # Get pattern mask for this character (0 if not in pattern)
        eq = peq.get(c, 0)

        # Compute diagonal: positions where we might have a match
        xv = eq | vn
        xh = (((eq & vp) + vp) ^ vp) | eq

        # Compute horizontal positive/negative
        hp = vn | ~(xh | vp)
        hn = xh & vp

        # Update score based on last row
        if hp & last_bit:
            score += 1
        if hn & last_bit:
            score -= 1

        # Shift HP and HN for next iteration
        hp = (hp << 1) | 1
        hn = hn << 1

        # Update VP and VN
        vp = hn | ~(xv | hp)
        vn = xv & hp

    return score


def _myers_block(pattern: List[str], text: List[str], word_size: int = 64) -> int:
    """
    Block-based Myers algorithm for patterns longer than word size.

    Divides the pattern into blocks of word_size bits and processes
    them together.

    Args:
        pattern: Pattern as list of characters
        text: Text as list of characters
        word_size: Size of machine word in bits

    Returns:
        Integer edit distance
    """
    m = len(pattern)
    n = len(text)

    # Number of blocks needed
    num_blocks = (m + word_size - 1) // word_size

    # Build pattern masks for each block
    peq_blocks: List[Dict[str, int]] = [{} for _ in range(num_blocks)]

    for i, c in enumerate(pattern):
        block_idx = i // word_size
        bit_pos = i % word_size

        if c not in peq_blocks[block_idx]:
            peq_blocks[block_idx][c] = 0
        peq_blocks[block_idx][c] |= (1 << bit_pos)

    # Initialize vectors for each block
    vp = [(1 << min(word_size, m - b * word_size)) - 1 for b in range(num_blocks)]
    vn = [0] * num_blocks

    score = m

    # Last bit mask for each block
    last_bits = []
    for b in range(num_blocks):
        block_size = min(word_size, m - b * word_size)
        last_bits.append(1 << (block_size - 1))

    # Process text
    for c in text:
        carry_hp = 1
        carry_hn = 0

        for b in range(num_blocks):
            eq = peq_blocks[b].get(c, 0)

            xv = eq | vn[b]
            xh_temp = eq & vp[b]
            xh = (xh_temp + vp[b] + carry_hp) ^ vp[b] | eq

            hp = vn[b] | ~(xh | vp[b])
            hn = xh & vp[b]

            # Carry out
            block_size = min(word_size, m - b * word_size)
            mask = (1 << block_size) - 1

            new_carry_hp = 1 if (hp & last_bits[b]) else 0
            new_carry_hn = 1 if (hn & last_bits[b]) else 0

            hp = ((hp << 1) | carry_hp) & mask
            hn = ((hn << 1) | carry_hn) & mask

            vp[b] = (hn | ~(xv | hp)) & mask
            vn[b] = (xv & hp) & mask

            carry_hp = new_carry_hp
            carry_hn = new_carry_hn

        # Update score using last block
        if carry_hp:
            score += 1
        if carry_hn:
            score -= 1

    return score


def myers_bitvector_with_trace(word1: str, word2: str) -> Tuple[int, List[int]]:
    """
    Compute edit distance and return the score trace.

    Returns the edit distance and a list showing the score after
    processing each character of the text.

    Args:
        word1: Pattern string
        word2: Text string

    Returns:
        Tuple of (final distance, list of scores after each text character)
    """
    pattern = list(word1)
    text = list(word2)

    m = len(pattern)
    n = len(text)

    if m == 0:
        return n, list(range(1, n + 1))
    if n == 0:
        return m, []

    # Build pattern masks
    peq: Dict[str, int] = {}
    for i, c in enumerate(pattern):
        if c not in peq:
            peq[c] = 0
        peq[c] |= (1 << i)

    vp = (1 << m) - 1
    vn = 0
    score = m
    last_bit = 1 << (m - 1)

    scores = []

    for c in text:
        eq = peq.get(c, 0)

        xv = eq | vn
        xh = (((eq & vp) + vp) ^ vp) | eq

        hp = vn | ~(xh | vp)
        hn = xh & vp

        if hp & last_bit:
            score += 1
        if hn & last_bit:
            score -= 1

        scores.append(score)

        hp = (hp << 1) | 1
        hn = hn << 1

        vp = hn | ~(xv | hp)
        vn = xv & hp

    return score, scores


def get_alphabet(word1: str, word2: str) -> Set[str]:
    """
    Get the combined alphabet of two strings.

    Args:
        word1: First string
        word2: Second string

    Returns:
        Set of unique characters
    """
    return set(word1) | set(word2)


if __name__ == "__main__":
    # Quick test
    test_cases = [
        ("kitten", "sitting", 3),
        ("празнк", "празник", 1),  # Missing и
        ("мчака", "мачка", 2),     # No transposition in Myers = 2
    ]

    print("Myers' Bit-Vector Algorithm Tests:")
    print("-" * 50)

    for w1, w2, expected in test_cases:
        result = myers_bitvector(w1, w2)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{w1}' → '{w2}': {result} (expected {expected})")

    # Show trace for educational purposes
    print("\nScore trace for 'kitten' → 'sitting':")
    score, trace = myers_bitvector_with_trace("kitten", "sitting")
    print(f"  Text chars: {list('sitting')}")
    print(f"  Scores:     {trace}")
    print(f"  Final:      {score}")
