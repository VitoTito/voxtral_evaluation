from dataclasses import dataclass
from typing import Optional

@dataclass
class EvaluationResult:
    """
    Represents the evaluation outcome of a single audio sample transcribed by the 
    Voxtral ASR model.

    This data structure captures key performance metrics for a single audio sample 
    during ASR evaluation. It includes the Word Error Rate (WER), detailed error 
    counts, both reference and predicted transcriptions, and the Inverse Real-Time 
    Factor (RTFx), which measures inference speed.

    Attributes
    ----------
    path : str
        File path to the audio sample that was evaluated.
    wer : Optional[float]
        Word Error Rate (WER), computed as: 
        (insertions + deletions + substitutions) / number of reference words.
        Value ranges from 0 (perfect transcription) to 1 (all words incorrect).
    insertions : Optional[int], default=None
        Number of insertion errors made by the model.
    deletions : Optional[int], default=None
        Number of deletion errors made by the model.
    substitutions : Optional[int], default=None
        Number of substitution errors made by the model.
    reference_length : Optional[int], default=None
        Total number of words in the reference transcription.
    prediction : str, default=""
        The text transcription predicted by the model.
    reference : str, default=""
        The ground truth or human-annotated reference transcription.
    audio_duration : Optional[float], default=None
        Audio duration in seconds.
    inference_time : Optional[float], default=None
        Inference time in seconds
    rtf_x : Optional[float], default=None
        Inverse Real-Time Factor (RTFx), defined as:
        (duration of audio in seconds) / (inference time in seconds).
        Values > 1.0 indicate faster-than-real-time inference.
    """
    path: str
    wer: Optional[float]
    insertions: Optional[int] = None
    deletions: Optional[int] = None
    substitutions: Optional[int] = None
    reference_length: Optional[int] = None
    prediction: str = ""
    reference: str = ""
    audio_duration: Optional[float] = None
    inference_time: Optional[float] = None
    rtf_x: Optional[float] = None 