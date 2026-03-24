"""
Unit tests for edit distance algorithms.

Tests all three algorithms (Levenshtein, Damerau-Levenshtein, Myers)
with both English (ASCII) and Macedonian (Cyrillic UTF-8) test cases.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spell_checker.algorithms import levenshtein, damerau_levenshtein, myers_bitvector
from spell_checker.algorithms.levenshtein import levenshtein_with_matrix, levenshtein_operations
from spell_checker.algorithms.damerau_levenshtein import damerau_levenshtein_operations
from spell_checker.algorithms.myers_bitvector import myers_bitvector_with_trace
from spell_checker.utils.unicode_helpers import (
    to_codepoints, from_codepoints, is_cyrillic, is_ascii_alpha, char_info
)


class TestLevenshtein(unittest.TestCase):
    """Test cases for Levenshtein distance algorithm."""

    def test_identical_strings(self):
        """Identical strings should have distance 0."""
        self.assertEqual(levenshtein("hello", "hello"), 0)
        self.assertEqual(levenshtein("", ""), 0)
        self.assertEqual(levenshtein("мачка", "мачка"), 0)

    def test_empty_strings(self):
        """Distance to/from empty string equals length of other string."""
        self.assertEqual(levenshtein("", "hello"), 5)
        self.assertEqual(levenshtein("hello", ""), 5)
        self.assertEqual(levenshtein("", "мачка"), 5)

    def test_single_insertion(self):
        """Single character insertion."""
        self.assertEqual(levenshtein("cat", "cats"), 1)
        self.assertEqual(levenshtein("мака", "мачка"), 1)

    def test_single_deletion(self):
        """Single character deletion."""
        self.assertEqual(levenshtein("cats", "cat"), 1)
        self.assertEqual(levenshtein("мачка", "мака"), 1)

    def test_single_substitution(self):
        """Single character substitution."""
        self.assertEqual(levenshtein("cat", "bat"), 1)
        self.assertEqual(levenshtein("мачка", "тачка"), 1)

    def test_english_classic_example(self):
        """Classic 'kitten' to 'sitting' example."""
        # kitten → sitten (substitute s for k) → sittin (substitute i for e) → sitting (insert g)
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_macedonian_missing_letter(self):
        """Macedonian: празнк → празник (missing 'и')."""
        self.assertEqual(levenshtein("празнк", "празник"), 1)

    def test_macedonian_transposition_requires_two_ops(self):
        """Macedonian: мчака → мачка requires 2 ops (no transposition in Levenshtein)."""
        # мчака vs мачка - positions 1,2 are swapped (ч,а vs а,ч)
        # Levenshtein needs 2 substitutions or delete+insert
        self.assertEqual(levenshtein("мчака", "мачка"), 2)

    def test_symmetry(self):
        """Levenshtein distance should be symmetric."""
        self.assertEqual(levenshtein("abc", "def"), levenshtein("def", "abc"))
        self.assertEqual(levenshtein("мачка", "куче"), levenshtein("куче", "мачка"))

    def test_unicode_codepoints_not_bytes(self):
        """Verify we work with code points, not UTF-8 bytes."""
        # Cyrillic 'а' is 2 bytes in UTF-8 but should be 1 character
        self.assertEqual(levenshtein("а", "б"), 1)  # Single substitution
        self.assertEqual(levenshtein("а", "аб"), 1)  # Single insertion


class TestDamerauLevenshtein(unittest.TestCase):
    """Test cases for Damerau-Levenshtein distance algorithm."""

    def test_identical_strings(self):
        """Identical strings should have distance 0."""
        self.assertEqual(damerau_levenshtein("hello", "hello"), 0)
        self.assertEqual(damerau_levenshtein("мачка", "мачка"), 0)

    def test_empty_strings(self):
        """Distance to/from empty string equals length of other string."""
        self.assertEqual(damerau_levenshtein("", "hello"), 5)
        self.assertEqual(damerau_levenshtein("hello", ""), 5)

    def test_single_insertion(self):
        """Single character insertion."""
        self.assertEqual(damerau_levenshtein("cat", "cats"), 1)

    def test_single_deletion(self):
        """Single character deletion."""
        self.assertEqual(damerau_levenshtein("cats", "cat"), 1)

    def test_single_substitution(self):
        """Single character substitution."""
        self.assertEqual(damerau_levenshtein("cat", "bat"), 1)

    def test_single_transposition(self):
        """Single adjacent transposition should be distance 1."""
        self.assertEqual(damerau_levenshtein("ab", "ba"), 1)
        self.assertEqual(damerau_levenshtein("teh", "the"), 1)

    def test_english_classic_example(self):
        """Classic 'kitten' to 'sitting' example."""
        self.assertEqual(damerau_levenshtein("kitten", "sitting"), 3)

    def test_macedonian_missing_letter(self):
        """Macedonian: празнк → празник (missing 'и')."""
        self.assertEqual(damerau_levenshtein("празнк", "празник"), 1)

    def test_macedonian_transposition_single_op(self):
        """Macedonian: мчака → мачка is single transposition (distance 1)."""
        # This is THE key differentiator from Levenshtein
        self.assertEqual(damerau_levenshtein("мчака", "мачка"), 1)

    def test_transposition_vs_levenshtein(self):
        """Damerau should be <= Levenshtein for all cases."""
        test_pairs = [
            ("ab", "ba"),
            ("teh", "the"),
            ("мчака", "мачка"),
            ("receieve", "receive"),
        ]
        for w1, w2 in test_pairs:
            dam = damerau_levenshtein(w1, w2)
            lev = levenshtein(w1, w2)
            self.assertLessEqual(dam, lev,
                f"Damerau ({dam}) should be <= Levenshtein ({lev}) for '{w1}'→'{w2}'")

    def test_symmetry(self):
        """Damerau-Levenshtein distance should be symmetric."""
        self.assertEqual(
            damerau_levenshtein("abc", "acb"),
            damerau_levenshtein("acb", "abc")
        )


class TestMyersBitvector(unittest.TestCase):
    """Test cases for Myers' bit-vector algorithm."""

    def test_identical_strings(self):
        """Identical strings should have distance 0."""
        self.assertEqual(myers_bitvector("hello", "hello"), 0)
        self.assertEqual(myers_bitvector("мачка", "мачка"), 0)

    def test_empty_strings(self):
        """Distance to/from empty string equals length of other string."""
        self.assertEqual(myers_bitvector("", "hello"), 5)
        self.assertEqual(myers_bitvector("hello", ""), 5)

    def test_single_insertion(self):
        """Single character insertion."""
        self.assertEqual(myers_bitvector("cat", "cats"), 1)

    def test_single_deletion(self):
        """Single character deletion."""
        self.assertEqual(myers_bitvector("cats", "cat"), 1)

    def test_single_substitution(self):
        """Single character substitution."""
        self.assertEqual(myers_bitvector("cat", "bat"), 1)

    def test_english_classic_example(self):
        """Classic 'kitten' to 'sitting' example."""
        self.assertEqual(myers_bitvector("kitten", "sitting"), 3)

    def test_macedonian_missing_letter(self):
        """Macedonian: празнк → празник (missing 'и')."""
        self.assertEqual(myers_bitvector("празнк", "празник"), 1)

    def test_macedonian_transposition_requires_two_ops(self):
        """Macedonian: мчака → мачка requires 2 ops (Myers = Levenshtein)."""
        # Myers computes Levenshtein distance, not Damerau
        self.assertEqual(myers_bitvector("мчака", "мачка"), 2)

    def test_matches_levenshtein(self):
        """Myers should give same results as Levenshtein."""
        test_pairs = [
            ("", ""),
            ("a", "a"),
            ("abc", "abc"),
            ("kitten", "sitting"),
            ("мачка", "куче"),
            ("празнк", "празник"),
            ("algorithm", "altruistic"),
        ]
        for w1, w2 in test_pairs:
            myers = myers_bitvector(w1, w2)
            lev = levenshtein(w1, w2)
            self.assertEqual(myers, lev,
                f"Myers ({myers}) should equal Levenshtein ({lev}) for '{w1}'→'{w2}'")

    def test_long_strings(self):
        """Test with strings longer than 64 characters (uses block algorithm)."""
        long1 = "a" * 100
        long2 = "a" * 100
        self.assertEqual(myers_bitvector(long1, long2), 0)

        long3 = "a" * 100
        long4 = "b" * 100
        self.assertEqual(myers_bitvector(long3, long4), 100)


class TestUnicodeHelpers(unittest.TestCase):
    """Test cases for Unicode helper functions."""

    def test_to_codepoints(self):
        """Test conversion to code points."""
        self.assertEqual(to_codepoints("abc"), [97, 98, 99])
        self.assertEqual(to_codepoints("мачка"), [1084, 1072, 1095, 1082, 1072])

    def test_from_codepoints(self):
        """Test conversion from code points."""
        self.assertEqual(from_codepoints([97, 98, 99]), "abc")
        self.assertEqual(from_codepoints([1084, 1072, 1095, 1082, 1072]), "мачка")

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        test_strings = ["hello", "мачка", "mixed текст 123"]
        for s in test_strings:
            self.assertEqual(from_codepoints(to_codepoints(s)), s)

    def test_is_cyrillic(self):
        """Test Cyrillic detection."""
        self.assertTrue(is_cyrillic("а"))
        self.assertTrue(is_cyrillic("я"))
        self.assertTrue(is_cyrillic("Б"))
        self.assertFalse(is_cyrillic("a"))
        self.assertFalse(is_cyrillic("1"))

    def test_is_ascii_alpha(self):
        """Test ASCII letter detection."""
        self.assertTrue(is_ascii_alpha("a"))
        self.assertTrue(is_ascii_alpha("Z"))
        self.assertFalse(is_ascii_alpha("1"))
        self.assertFalse(is_ascii_alpha("а"))  # Cyrillic 'а'

    def test_char_info(self):
        """Test character info function."""
        info = char_info("м")
        self.assertEqual(info['codepoint'], 1084)
        self.assertEqual(info['hex'], 'U+043C')
        self.assertTrue(info['is_cyrillic'])
        self.assertFalse(info['is_ascii'])


class TestAllAlgorithmsComparison(unittest.TestCase):
    """
    Comprehensive comparison tests ensuring all algorithms behave correctly
    and differences are as expected.
    """

    def test_required_english_example(self):
        """Required test: kitten → sitting = 3 for all algorithms."""
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(damerau_levenshtein("kitten", "sitting"), 3)
        self.assertEqual(myers_bitvector("kitten", "sitting"), 3)

    def test_required_macedonian_insertion(self):
        """Required test: празнк → празник = 1 for all algorithms."""
        self.assertEqual(levenshtein("празнк", "празник"), 1)
        self.assertEqual(damerau_levenshtein("празнк", "празник"), 1)
        self.assertEqual(myers_bitvector("празнк", "празник"), 1)

    def test_required_macedonian_transposition(self):
        """Required test: мчака → мачка - Lev=2, Dam=1, Myers=2."""
        # This is the KEY TEST that shows the difference
        lev = levenshtein("мчака", "мачка")
        dam = damerau_levenshtein("мчака", "мачка")
        myers = myers_bitvector("мчака", "мачка")

        self.assertEqual(lev, 2, "Levenshtein should be 2 (no transposition)")
        self.assertEqual(dam, 1, "Damerau should be 1 (has transposition)")
        self.assertEqual(myers, 2, "Myers should be 2 (same as Levenshtein)")

    def test_english_transposition_typos(self):
        """Common English transposition typos."""
        # "teh" → "the"
        self.assertEqual(levenshtein("teh", "the"), 2)
        self.assertEqual(damerau_levenshtein("teh", "the"), 1)
        self.assertEqual(myers_bitvector("teh", "the"), 2)

        # "recieve" → "receive"
        self.assertEqual(damerau_levenshtein("recieve", "receive"), 1)

    def test_algorithms_relationship(self):
        """Verify: Damerau <= Levenshtein = Myers for all cases."""
        test_pairs = [
            ("ab", "ba"),
            ("abc", "acb"),
            ("hello", "hlelo"),
            ("мчака", "мачка"),
            ("teh", "the"),
        ]

        for w1, w2 in test_pairs:
            lev = levenshtein(w1, w2)
            dam = damerau_levenshtein(w1, w2)
            myers = myers_bitvector(w1, w2)

            self.assertEqual(lev, myers,
                f"Levenshtein ({lev}) should equal Myers ({myers}) for '{w1}'→'{w2}'")
            self.assertLessEqual(dam, lev,
                f"Damerau ({dam}) should be <= Levenshtein ({lev}) for '{w1}'→'{w2}'")


class TestEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_single_character(self):
        """Single character strings."""
        self.assertEqual(levenshtein("a", "b"), 1)
        self.assertEqual(levenshtein("a", "a"), 0)
        self.assertEqual(levenshtein("м", "н"), 1)

    def test_repeated_characters(self):
        """Strings with repeated characters."""
        self.assertEqual(levenshtein("aaa", "aaaa"), 1)
        self.assertEqual(levenshtein("aaa", "bbb"), 3)

    def test_completely_different(self):
        """Completely different strings."""
        self.assertEqual(levenshtein("abc", "xyz"), 3)
        # мачка (5 chars) → куче (4 chars): they share 'ч' and 'к', distance is 4
        self.assertEqual(levenshtein("мачка", "куче"), 4)

    def test_prefix(self):
        """One string is prefix of another."""
        self.assertEqual(levenshtein("test", "testing"), 3)
        self.assertEqual(levenshtein("мач", "мачка"), 2)

    def test_suffix(self):
        """One string is suffix of another."""
        self.assertEqual(levenshtein("ing", "testing"), 4)

    def test_unicode_normalization(self):
        """Test that unicode normalization works correctly."""
        # These should be treated as equivalent after normalization
        # (though our algorithms work on raw characters)
        self.assertEqual(levenshtein("café", "cafe"), 1)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
