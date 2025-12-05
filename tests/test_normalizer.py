"""
Unit tests for the TextNormalizer class.

Checks that text normalization correctly handles:
- Lowercasing
- Punctuation removal
- Whitespace normalization
- Simple time expressions like '5h' -> '5 h'
"""

import unittest
from src.preprocessing.normalizer import TextNormalizer


class TestTextNormalizer(unittest.TestCase):
    """
    Test cases for the TextNormalizer class.
    """

    def setUp(self):
        """
        Initialize a TextNormalizer instance before each test.
        """
        self.normalizer = TextNormalizer()

    def test_basic_lowercase_and_punctuation(self):
        """
        Check that normalization lowercases the text and removes punctuation.
        """
        text = "Bonjour, Comment ça va?"
        expected = "bonjour comment ça va"
        self.assertEqual(self.normalizer.normalize(text), expected)

    def test_time_normalization(self):
        """
        Check that time expressions like '5h' are normalized to '5 h'.
        """
        text = "Rendez-vous à 5h et non à cinq heures"
        expected = "rendezvous à 5 h et non à cinq heures"
        self.assertEqual(self.normalizer.normalize(text), expected)


if __name__ == "__main__":
    unittest.main()
