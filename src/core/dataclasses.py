from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
    """
    Stores the evaluation result for a single audio sample.
    """
    path: str
    prediction: str
    reference: str
    wer: Optional[float] = None
    cer: Optional[float] = None
