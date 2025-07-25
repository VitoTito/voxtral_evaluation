import re
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, RemoveWhiteSpace

class TextNormalizer:
    """
    Normalize French text for ASR evaluation with Voxtral:
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    - Convert French number words to digits
    - Normalize time expressions like "5h", "cinq heures" -> "5 h"
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
        Normalize a text string by converting French number words and applying standard cleaning.

        Parameters
        ----------
        text : str
            Input raw text.

        Returns
        -------
        str
            Normalized text.
        """
        text = self.convert_numbers(text)
        return self.pipeline(text)

    def convert_numbers(self, text: str) -> str:
        """
        Convert French number words in the text to digits.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Text with French number words replaced by digits.
        """
        return self._replace_french_numbers(text)

    def _replace_french_numbers(self, text: str) -> str:
        """
        Replace French number words (units, teens, tens, hundreds) with their digit equivalents.

        Handles compound numbers with hyphens and special cases like 'et'.

        Parameters
        ----------
        text : str
            Input text containing French number words.

        Returns
        -------
        str
            Text with French number words replaced by digits.
        """
        units = {
            "zéro": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
            "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9
        }
        teens = {
            "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14,
            "quinze": 15, "seize": 16, "dix-sept": 17, "dix-huit": 18, "dix-neuf": 19
        }
        tens = {
            "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50,
            "soixante": 60, "soixante-dix": 70, "quatre-vingt": 80, "quatre-vingt-dix": 90
        }
        hundreds = {"cent": 100}

        number_words = list(units.keys()) + list(teens.keys()) + list(tens.keys()) + list(hundreds.keys()) + [
            "et", "vingt-et-un", "trente-et-un", "quatre-vingt-onze"
        ]
        pattern = re.compile(
            r'\b(?:' + '|'.join(re.escape(word) for word in number_words) + r')(?:-(?:' + '|'.join(re.escape(word) for word in number_words) + r'))*\b',
            flags=re.IGNORECASE
        )

        def word_to_number(match):
            """
            Convert a matched French number word sequence into its integer numeric representation.

            Parameters
            ----------
            match : re.Match
                A regex match object containing the matched French number words.

            Returns
            -------
            str
                The numeric value as a string corresponding to the matched French number words.
            """
            # Get the matched text, convert to lowercase, replace hyphens with spaces
            chunk = match.group(0).lower().replace("-", " ")
            words = chunk.split()
            total = 0
            current = 0

            for word in words:
                if word in units:
                    current += units[word]
                elif word in teens:
                    current += teens[word]
                elif word in tens:
                    current += tens[word]
                elif word == "et":
                    continue
                elif word in hundreds:
                    if current == 0:
                        current = 1
                    current *= hundreds[word]
                    total += current
                    current = 0
            total += current
            return str(total)

        text = pattern.sub(word_to_number, text)

        # Normalize hour expressions like "5h", "cinq heures" → "5 h"
        text = re.sub(r'\b(\d+)\s?h(?:eures?)?\b', r'\1 h', text)

        return text