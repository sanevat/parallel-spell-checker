"""
Spell Checker with Multiple Edit Distance Algorithms

A research project comparing Levenshtein, Damerau-Levenshtein, and
Myers' bit-vector algorithms for spell checking, with support for
both ASCII (English) and UTF-8 (Macedonian Cyrillic) text.
"""

from .algorithms import levenshtein, damerau_levenshtein, myers_bitvector
from .spell_checker import spell_check, SpellChecker

__version__ = "0.1.0"
__all__ = [
    'levenshtein',
    'damerau_levenshtein',
    'myers_bitvector',
    'spell_check',
    'SpellChecker'
]
