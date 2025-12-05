"""
TextNormalizer module.

Provides a simple text normalization class for French ASR evaluation,
including lowercasing, punctuation removal, whitespace normalization,
and simple time expression normalization.
"""


import re
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, RemoveWhiteSpace


class TextNormalizer:
    """
    Normalize French text for ASR evaluation.

    The pipeline performs:
    - Lowercasing
    - Removing punctuation
    - Normalizing whitespace
    - Normalizing simple time expressions like "5h" -> "5 h"
    """

    def __init__(self):
        """
        Initialize the text normalization pipeline with jiwer transforms.
        """
        self.pipeline = Compose([
            ToLowerCase(),
            RemovePunctuation(),
            RemoveMultipleSpaces(),
            RemoveWhiteSpace(replace_by_space=True),
        ])

    def normalize(self, text: str) -> str:
        """
        Normalize a text string for ASR evaluation.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Normalized text.
        """
        text = self._normalize_time_expressions(text)
        return self.pipeline(text)

    def _normalize_time_expressions(self, text: str) -> str:
        """
        Normalize time expressions like '5h', '5 heures' -> '5 h'.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Text with normalized hour expressions.
        """
        text = re.sub(r'\b(\d+)\s?h(?:eures?)?\b', r'\1 h', text, flags=re.IGNORECASE)
        return text
