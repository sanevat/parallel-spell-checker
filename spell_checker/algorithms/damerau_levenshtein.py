"""
Damerau-Levenshtein Edit Distance Algorithm

Extension of Levenshtein that also considers transposition of two adjacent
characters as a single edit operation. This is particularly useful for
spell checking as transposition errors are very common in typing.

Operations: insertion, deletion, substitution, transposition

Time Complexity: O(m * n)
Space Complexity: O(m * n) for full algorithm

Works with Unicode code points for proper Cyrillic/UTF-8 support.
"""

from typing import List, Tuple, Optional, Dict


def damerau_levenshtein(word1: str, word2: str) -> int:
    """
    Compute the Damerau-Levenshtein distance between two strings.

    The Damerau-Levenshtein distance is the minimum number of operations
    (insertions, deletions, substitutions, and transpositions of adjacent
    characters) required to change word1 into word2.

    This is the "optimal string alignment" variant which does not allow
    multiple edits on the same substring.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        Integer edit distance

    Examples:
        >>> damerau_levenshtein("kitten", "sitting")
        3
        >>> damerau_levenshtein("празнк", "празник")  # Missing 'и'
        1
        >>> damerau_levenshtein("мчака", "мачка")  # Transposition: ча → ач
        1
    """
    # Work with Unicode code points
    s1 = list(word1)
    s2 = list(word2)

    m, n = len(s1), len(s2)

    # Edge cases
    if m == 0:
        return n
    if n == 0:
        return m

    # Create DP matrix
    # We need (m+1) x (n+1) matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

            # Check for transposition
            if (i > 1 and j > 1 and
                s1[i - 1] == s2[j - 2] and
                s1[i - 2] == s2[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)  # Transposition

    return dp[m][n]


def damerau_levenshtein_unrestricted(word1: str, word2: str) -> int:
    """
    Compute the true Damerau-Levenshtein distance (unrestricted).

    This version allows multiple edits on the same substring (e.g., transposing
    and then inserting into the transposed characters). Uses more memory but
    gives the true minimum edit distance.

    Uses the algorithm with "DA" (last occurrence) tracking.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        Integer edit distance
    """
    s1 = list(word1)
    s2 = list(word2)

    m, n = len(s1), len(s2)

    # Edge cases
    if m == 0:
        return n
    if n == 0:
        return m

    # Build alphabet of all unique characters
    alphabet: Dict[str, int] = {}
    for c in s1 + s2:
        if c not in alphabet:
            alphabet[c] = len(alphabet)

    # Initialize DP table with extra row and column for the "infinite" boundary
    max_dist = m + n
    dp = [[0] * (n + 2) for _ in range(m + 2)]

    dp[0][0] = max_dist

    for i in range(0, m + 1):
        dp[i + 1][0] = max_dist
        dp[i + 1][1] = i

    for j in range(0, n + 1):
        dp[0][j + 1] = max_dist
        dp[1][j + 1] = j

    # Track last row where each character was seen
    da: Dict[str, int] = {c: 0 for c in alphabet}

    for i in range(1, m + 1):
        db = 0  # Last column where s1[i] matched in s2

        for j in range(1, n + 1):
            i1 = da.get(s2[j - 1], 0)
            j1 = db

            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            if cost == 0:
                db = j

            dp[i + 1][j + 1] = min(
                dp[i][j] + cost,           # Substitution (or match)
                dp[i + 1][j] + 1,          # Insertion
                dp[i][j + 1] + 1,          # Deletion
                dp[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1)  # Transposition
            )

        da[s1[i - 1]] = i

    return dp[m + 1][n + 1]


def damerau_levenshtein_with_matrix(word1: str, word2: str) -> Tuple[int, List[List[int]]]:
    """
    Compute Damerau-Levenshtein distance and return the full DP matrix.

    Useful for debugging and visualization.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        Tuple of (distance, full DP matrix)
    """
    s1 = list(word1)
    s2 = list(word2)

    m, n = len(s1), len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

            if (i > 1 and j > 1 and
                s1[i - 1] == s2[j - 2] and
                s1[i - 2] == s2[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[m][n], dp


def damerau_levenshtein_operations(word1: str, word2: str) -> List[Tuple[str, Optional[str], Optional[str], int]]:
    """
    Compute Damerau-Levenshtein distance and return the sequence of operations.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        List of operations: (operation_type, old_char(s), new_char(s), position)
        operation_type is one of: 'match', 'substitute', 'insert', 'delete', 'transpose'
    """
    s1 = list(word1)
    s2 = list(word2)

    m, n = len(s1), len(s2)

    _, dp = damerau_levenshtein_with_matrix(word1, word2)

    operations = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            operations.append(('match', s1[i - 1], s2[j - 1], i - 1))
            i -= 1
            j -= 1
        elif (i > 1 and j > 1 and
              s1[i - 1] == s2[j - 2] and
              s1[i - 2] == s2[j - 1] and
              dp[i][j] == dp[i - 2][j - 2] + 1):
            # Transposition
            operations.append(('transpose', s1[i - 2] + s1[i - 1], s2[j - 2] + s2[j - 1], i - 2))
            i -= 2
            j -= 2
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            operations.append(('substitute', s1[i - 1], s2[j - 1], i - 1))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            operations.append(('insert', None, s2[j - 1], i))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append(('delete', s1[i - 1], None, i - 1))
            i -= 1
        else:
            # Fallback for match (shouldn't normally reach here)
            if i > 0 and j > 0:
                operations.append(('match', s1[i - 1], s2[j - 1], i - 1))
                i -= 1
                j -= 1
            elif j > 0:
                operations.append(('insert', None, s2[j - 1], i))
                j -= 1
            else:
                operations.append(('delete', s1[i - 1], None, i - 1))
                i -= 1

    operations.reverse()
    return operations


if __name__ == "__main__":
    # Quick test
    test_cases = [
        ("kitten", "sitting", 3),
        ("празнк", "празник", 1),  # Missing и
        ("мчака", "мачка", 1),     # Transposition (Dam can do it in 1!)
    ]

    print("Damerau-Levenshtein Distance Tests:")
    print("-" * 50)

    for w1, w2, expected in test_cases:
        result = damerau_levenshtein(w1, w2)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{w1}' → '{w2}': {result} (expected {expected})")

    # Show the transposition operation
    print("\nTransposition example:")
    ops = damerau_levenshtein_operations("мчака", "мачка")
    for op in ops:
        if op[0] != 'match':
            print(f"  {op[0]}: '{op[1]}' → '{op[2]}' at position {op[3]}")
