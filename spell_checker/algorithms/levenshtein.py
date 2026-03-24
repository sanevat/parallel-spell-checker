"""
Levenshtein Edit Distance Algorithm

The classic dynamic programming algorithm for computing the minimum number
of single-character edits (insertions, deletions, substitutions) needed
to transform one string into another.

Time Complexity: O(m * n)
Space Complexity: O(min(m, n)) with optimization

Works with Unicode code points for proper Cyrillic/UTF-8 support.
"""

from typing import List, Tuple, Optional


def levenshtein(word1: str, word2: str) -> int:
    """
    Compute the Levenshtein distance between two strings.

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, substitutions) required to change word1 into word2.

    This implementation works with Unicode code points, not bytes, ensuring
    correct behavior with Cyrillic and other non-ASCII characters.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        Integer edit distance

    Examples:
        >>> levenshtein("kitten", "sitting")
        3
        >>> levenshtein("празнк", "празник")  # Missing 'и' in Macedonian
        1
    """
    # Work with Unicode code points, not bytes
    s1 = list(word1)
    s2 = list(word2)

    m, n = len(s1), len(s2)

    # Optimize: use shorter string for columns (space optimization)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    # Edge cases
    if n == 0:
        return m

    # Use two rows for space optimization: O(min(m,n)) instead of O(m*n)
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i

        for j in range(1, n + 1):
            # Cost is 0 if characters match, 1 otherwise
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            curr_row[j] = min(
                prev_row[j] + 1,      # Deletion
                curr_row[j - 1] + 1,  # Insertion
                prev_row[j - 1] + cost  # Substitution
            )

        # Swap rows
        prev_row, curr_row = curr_row, prev_row

    return prev_row[n]


def levenshtein_with_matrix(word1: str, word2: str) -> Tuple[int, List[List[int]]]:
    """
    Compute Levenshtein distance and return the full DP matrix.

    Useful for debugging, visualization, and backtracking to find
    the actual edit operations.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        Tuple of (distance, full DP matrix)
    """
    s1 = list(word1)
    s2 = list(word2)

    m, n = len(s1), len(s2)

    # Create full matrix
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
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[m][n], dp


def levenshtein_operations(word1: str, word2: str) -> List[Tuple[str, Optional[str], Optional[str], int]]:
    """
    Compute Levenshtein distance and return the sequence of edit operations.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        List of operations: (operation_type, old_char, new_char, position)
        operation_type is one of: 'match', 'substitute', 'insert', 'delete'
    """
    s1 = list(word1)
    s2 = list(word2)

    m, n = len(s1), len(s2)

    # Build full matrix for backtracking
    _, dp = levenshtein_with_matrix(word1, word2)

    # Backtrack to find operations
    operations = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            # Characters match
            operations.append(('match', s1[i - 1], s2[j - 1], i - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            operations.append(('substitute', s1[i - 1], s2[j - 1], i - 1))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # Insertion
            operations.append(('insert', None, s2[j - 1], i))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion
            operations.append(('delete', s1[i - 1], None, i - 1))
            i -= 1

    operations.reverse()
    return operations


if __name__ == "__main__":
    # Quick test
    test_cases = [
        ("kitten", "sitting", 3),
        ("празнк", "празник", 1),  # Missing и
        ("мчака", "мачка", 2),     # Transposition (Lev can't do it in 1)
    ]

    print("Levenshtein Distance Tests:")
    print("-" * 50)

    for w1, w2, expected in test_cases:
        result = levenshtein(w1, w2)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{w1}' → '{w2}': {result} (expected {expected})")
