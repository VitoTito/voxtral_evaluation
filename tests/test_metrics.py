"""
Unit tests for the WER and CER metric functions.

Ensures that the word error rate (WER) and character error rate (CER) are computed correctly
on simple, controlled examples.
"""

import unittest
from src.metrics.wer import compute_wer
from src.metrics.cer import compute_cer


class TestMetrics(unittest.TestCase):
    """
    Test cases for metric computations.
    """

    def test_wer_simple(self):
        """
        Verify that compute_wer correctly computes the WER for a simple sentence with one deletion.
        """
        ref = "bonjour comment ça va"
        hyp = "bonjour comment va"
        self.assertAlmostEqual(compute_wer(ref, hyp), 1/4)

    def test_cer_simple(self):
        """
        Verify that compute_cer correctly computes the CER for a simple string with one missing character.
        """
        ref = "abc"
        hyp = "ac"
        self.assertAlmostEqual(compute_cer(ref, hyp), 1/3)


if __name__ == "__main__":
    unittest.main()