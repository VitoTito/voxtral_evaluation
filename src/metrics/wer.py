"""
WER (Word Error Rate) metric for ASR evaluation.

Calculates the Word Error Rate (WER) between a reference text and a predicted text
using the standard formula:

    WER = (insertions + deletions + substitutions) / total reference words

Dependencies:
- jiwer
"""

from jiwer import wer


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute the WER between reference and hypothesis strings.

    Parameters
    ----------
    reference : str
        Ground-truth transcription.
    hypothesis : str
        Predicted transcription.

    Returns
    -------
    float
        Word Error Rate (WER), between 0 (perfect) and 1 (all words wrong).
    """
    return wer(reference, hypothesis)
