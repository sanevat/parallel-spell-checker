"""
Main Spell Checker Module

Provides spell checking functionality using multiple edit distance algorithms.
Supports both English (ASCII) and Macedonian (Cyrillic UTF-8) text.
"""

from typing import List, Tuple, Dict, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum

from .algorithms import levenshtein, damerau_levenshtein, myers_bitvector
from .utils.unicode_helpers import normalize_text, get_script


class Algorithm(Enum):
    """Available edit distance algorithms."""
    LEVENSHTEIN = "levenshtein"
    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    MYERS = "myers"


@dataclass
class SpellCheckResult:
    """Result of a spell check operation."""
    word: str
    is_correct: bool
    suggestions: List[Tuple[str, int]]  # (word, distance) pairs
    algorithm: Algorithm
    dictionary_size: int


def spell_check(
    word: str,
    dictionary: List[str],
    algorithm: Algorithm = Algorithm.LEVENSHTEIN,
    max_distance: int = 3,
    max_suggestions: int = 5,
    normalize: bool = True
) -> SpellCheckResult:
    """
    Check a word against a dictionary and return suggestions.

    Args:
        word: Word to check
        dictionary: List of valid words
        algorithm: Which edit distance algorithm to use
        max_distance: Maximum edit distance for suggestions
        max_suggestions: Maximum number of suggestions to return
        normalize: Whether to normalize case/whitespace

    Returns:
        SpellCheckResult with suggestions sorted by distance

    Examples:
        >>> result = spell_check("teh", ["the", "tea", "ten"])
        >>> result.suggestions[0]
        ('the', 1)
    """
    # Select distance function
    if algorithm == Algorithm.LEVENSHTEIN:
        distance_fn = levenshtein
    elif algorithm == Algorithm.DAMERAU_LEVENSHTEIN:
        distance_fn = damerau_levenshtein
    elif algorithm == Algorithm.MYERS:
        distance_fn = myers_bitvector
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Normalize if requested
    check_word = normalize_text(word) if normalize else word

    # Check if word is in dictionary
    dict_set = set(normalize_text(w) if normalize else w for w in dictionary)
    is_correct = check_word in dict_set

    if is_correct:
        return SpellCheckResult(
            word=word,
            is_correct=True,
            suggestions=[(word, 0)],
            algorithm=algorithm,
            dictionary_size=len(dictionary)
        )

    # Find suggestions
    suggestions = []
    for dict_word in dictionary:
        check_dict_word = normalize_text(dict_word) if normalize else dict_word
        distance = distance_fn(check_word, check_dict_word)

        if distance <= max_distance:
            suggestions.append((dict_word, distance))

    # Sort by distance, then alphabetically
    suggestions.sort(key=lambda x: (x[1], x[0]))

    # Limit suggestions
    suggestions = suggestions[:max_suggestions]

    return SpellCheckResult(
        word=word,
        is_correct=False,
        suggestions=suggestions,
        algorithm=algorithm,
        dictionary_size=len(dictionary)
    )


def spell_check_all_algorithms(
    word: str,
    dictionary: List[str],
    max_distance: int = 3,
    max_suggestions: int = 5,
    normalize: bool = True
) -> Dict[Algorithm, SpellCheckResult]:
    """
    Check a word using all three algorithms and compare results.

    Args:
        word: Word to check
        dictionary: List of valid words
        max_distance: Maximum edit distance for suggestions
        max_suggestions: Maximum number of suggestions to return
        normalize: Whether to normalize case/whitespace

    Returns:
        Dictionary mapping algorithm to SpellCheckResult
    """
    results = {}
    for algo in Algorithm:
        results[algo] = spell_check(
            word, dictionary, algo, max_distance, max_suggestions, normalize
        )
    return results


class SpellChecker:
    """
    A spell checker class with configurable algorithm and dictionary.

    Provides an object-oriented interface for spell checking with
    persistent dictionary and settings.
    """

    def __init__(
        self,
        dictionary: Optional[List[str]] = None,
        algorithm: Algorithm = Algorithm.LEVENSHTEIN,
        max_distance: int = 3,
        max_suggestions: int = 5,
        normalize: bool = True
    ):
        """
        Initialize the spell checker.

        Args:
            dictionary: Initial dictionary of valid words
            algorithm: Default algorithm to use
            max_distance: Maximum edit distance for suggestions
            max_suggestions: Maximum number of suggestions to return
            normalize: Whether to normalize words
        """
        self.dictionary: List[str] = dictionary or []
        self._dict_set: Set[str] = set()
        self.algorithm = algorithm
        self.max_distance = max_distance
        self.max_suggestions = max_suggestions
        self.normalize = normalize

        self._rebuild_dict_set()

    def _rebuild_dict_set(self) -> None:
        """Rebuild the internal dictionary set for fast lookups."""
        if self.normalize:
            self._dict_set = set(normalize_text(w) for w in self.dictionary)
        else:
            self._dict_set = set(self.dictionary)

    def add_word(self, word: str) -> None:
        """Add a word to the dictionary."""
        self.dictionary.append(word)
        normalized = normalize_text(word) if self.normalize else word
        self._dict_set.add(normalized)

    def add_words(self, words: List[str]) -> None:
        """Add multiple words to the dictionary."""
        for word in words:
            self.add_word(word)

    def load_dictionary(self, filepath: str, encoding: str = 'utf-8') -> int:
        """
        Load dictionary from a file (one word per line).

        Args:
            filepath: Path to dictionary file
            encoding: File encoding (default: utf-8)

        Returns:
            Number of words loaded
        """
        with open(filepath, 'r', encoding=encoding) as f:
            words = [line.strip() for line in f if line.strip()]

        self.dictionary = words
        self._rebuild_dict_set()
        return len(words)

    def is_correct(self, word: str) -> bool:
        """Check if a word is spelled correctly."""
        check_word = normalize_text(word) if self.normalize else word
        return check_word in self._dict_set

    def check(self, word: str) -> SpellCheckResult:
        """Check a word and get suggestions."""
        return spell_check(
            word,
            self.dictionary,
            self.algorithm,
            self.max_distance,
            self.max_suggestions,
            self.normalize
        )

    def check_all_algorithms(self, word: str) -> Dict[Algorithm, SpellCheckResult]:
        """Check a word using all algorithms."""
        return spell_check_all_algorithms(
            word,
            self.dictionary,
            self.max_distance,
            self.max_suggestions,
            self.normalize
        )

    def correct(self, word: str) -> str:
        """
        Get the best correction for a word.

        Returns the original word if correct, otherwise the closest match.
        """
        if self.is_correct(word):
            return word

        result = self.check(word)
        if result.suggestions:
            return result.suggestions[0][0]
        return word

    def check_text(self, text: str) -> List[Tuple[str, SpellCheckResult]]:
        """
        Check all words in a text.

        Args:
            text: Input text with words separated by whitespace

        Returns:
            List of (word, SpellCheckResult) tuples for misspelled words
        """
        words = text.split()
        misspelled = []

        for word in words:
            # Strip punctuation for checking
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word and not self.is_correct(clean_word):
                result = self.check(clean_word)
                misspelled.append((word, result))

        return misspelled


# Sample dictionaries for testing
SAMPLE_ENGLISH_DICT = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "kitten", "sitting", "cat", "dog", "hello", "world", "spell", "check",
    "algorithm", "distance", "edit", "word", "text", "test", "example"
]

SAMPLE_MACEDONIAN_DICT = [
    "празник", "мачка", "куче", "дома", "работа", "книга", "вода", "храна",
    "добар", "лош", "голем", "мал", "нов", "стар", "убав", "грд",
    "јас", "ти", "тој", "таа", "ние", "вие", "тие", "што", "кој", "каде",
    "денес", "утре", "вчера", "сега", "потоа", "секогаш", "никогаш"
]


if __name__ == "__main__":
    # Demo
    print("Spell Checker Demo")
    print("=" * 60)

    # English example
    checker = SpellChecker(SAMPLE_ENGLISH_DICT)

    print("\nEnglish spell check:")
    for word in ["kitten", "kiten", "sittin"]:
        result = checker.check(word)
        if result.is_correct:
            print(f"  '{word}' - correct")
        else:
            print(f"  '{word}' - suggestions: {result.suggestions}")

    # Macedonian example
    checker_mk = SpellChecker(SAMPLE_MACEDONIAN_DICT)

    print("\nMacedonian spell check:")
    for word in ["мачка", "мчака", "празнк"]:
        result = checker_mk.check(word)
        if result.is_correct:
            print(f"  '{word}' - correct")
        else:
            print(f"  '{word}' - suggestions: {result.suggestions}")

    # Compare algorithms on transposition
    print("\nComparing algorithms on 'мчака' → 'мачка' (transposition):")
    results = checker_mk.check_all_algorithms("мчака")
    for algo, result in results.items():
        if result.suggestions:
            best = result.suggestions[0]
            print(f"  {algo.value}: '{best[0]}' (distance={best[1]})")
