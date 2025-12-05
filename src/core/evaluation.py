from typing import List
import csv
from src.core.dataclasses import EvaluationResult
from src.evaluators.voxtral import VoxtralEvaluator
from src.preprocessing.normalizer import TextNormalizer
from src.metrics import compute_wer, compute_cer

class EvaluatorPipeline:
    """
    Orchestrates batch evaluation of audio files using VoxtralEvaluator.
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Parameters
        ----------
        model_path : str
            Path to the Voxtral model.
        device : str, optional
            Device to run the model on ('cuda' or 'cpu'). Defaults to auto-detect.
        """
        self.evaluator = VoxtralEvaluator(model_path=model_path, device=device)
        self.normalizer = TextNormalizer()

    def evaluate_all(self, annotations_path: str) -> List[EvaluationResult]:
        """
        Evaluate all audio files listed in the CSV annotations file.

        Parameters
        ----------
        annotations_path : str
            Path to a CSV file containing columns 'path' and 'reference'.

        Returns
        -------
        List[EvaluationResult]
            List of evaluation results for each audio sample.
        """
        results = []

        with open(annotations_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = row["path"]
                reference = self.normalizer.normalize(row["reference"])
                prediction = self.normalizer.normalize(self.evaluator.transcribe(audio_path))

                wer = compute_wer(reference, prediction)
                cer = compute_cer(reference, prediction)

                results.append(
                    EvaluationResult(
                        path=audio_path,
                        prediction=prediction,
                        reference=row["reference"],
                        wer=wer,
                        cer=cer
                    )
                )

        return results