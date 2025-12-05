"""
CER (Character Error Rate) metric for ASR evaluation.

Calculates the Character Error Rate (CER) between a reference text and a predicted text
using the standard formula:

    CER = (insertions + deletions + substitutions) / total characters in reference

Dependencies:
- jiwer
"""

from jiwer import cer


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute the CER between reference and hypothesis strings.

    Parameters
    ----------
    reference : str
        Ground-truth transcription.
    hypothesis : str
        Predicted transcription.

    Returns
    -------
    float
        Character Error Rate (CER), between 0 (perfect) and 1 (all chars wrong).
    """
    return cer(reference, hypothesis)
