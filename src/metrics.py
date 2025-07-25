from typing import List, Dict
import numpy as np
import pandas as pd
from dataclasses import asdict
from jiwer import process_words

from src.dataclasses import EvaluationResult
from src.normalizer import TextNormalizer


class VoxtralMetrics:
    """
    Encapsulates evaluation metrics for ASR systems including:

    - Word Error Rate (WER) and related error counts (insertions, deletions, substitutions).
    - RTFx (inverse Real-Time Factor) evaluation.
    - Descriptive statistics computation and error reporting tools.
    """

    def __init__(self):
        """
        Initialize the metrics module with a text normalizer.
        """
        self.normalizer = TextNormalizer()


    def compute_statistics(self, results: List[EvaluationResult]) -> tuple[Dict, Dict]:
        """
        Compute descriptive statistics for WER and RTFx based on evaluation results.

        Parameters
        ----------
        results : List[EvaluationResult]
            List of individual evaluation results to aggregate.

        Returns
        -------
        Tuple[Dict, Dict]
            - Dictionary of WER statistics including weighted WER.
            - Dictionary of RTFx statistics including global RTFx.
        """
        valid_wer_samples = [
            r for r in results if r.wer is not None and r.reference_length
        ]
        valid_rtf_samples = [
            r for r in results if r.rtf_x is not None and r.rtf_x > 0 and r.reference_length
        ]

        # WER aggregation
        wer_values = np.array([r.wer for r in valid_wer_samples]) if valid_wer_samples else np.array([])
        total_insertions = sum(r.insertions for r in valid_wer_samples if r.insertions is not None)
        total_deletions = sum(r.deletions for r in valid_wer_samples if r.deletions is not None)
        total_substitutions = sum(r.substitutions for r in valid_wer_samples if r.substitutions is not None)
        total_ref_words = sum(r.reference_length for r in valid_wer_samples)

        weighted_wer = (
            (total_insertions + total_deletions + total_substitutions) / total_ref_words
            if total_ref_words > 0
            else None
        )

        print("\nDescriptive Statistics - WER:")
        if weighted_wer is not None:
            print(f"  Weighted WER:   {weighted_wer:.2%}")
        if wer_values.size > 0:
            print(f"  Mean:           {wer_values.mean():.2%}")
            print(f"  Median:         {np.median(wer_values):.2%}")
            print(f"  Std deviation:  {wer_values.std():.2%}")
            print(f"  Variance:       {wer_values.var():.4f}")
            print(f"  1st Quartile:   {np.percentile(wer_values, 25):.2%}")
            print(f"  3rd Quartile:   {np.percentile(wer_values, 75):.2%}")
            print(f"  Min:            {wer_values.min():.2%}")
            print(f"  Max:            {wer_values.max():.2%}")
            print(f"  Count:          {len(wer_values)}")
        else:
            print("  No valid WER values.")

        # RTFx aggregation
        rtf_values = np.array([r.rtf_x for r in valid_rtf_samples]) if valid_rtf_samples else np.array([])
        total_audio_duration = sum(r.audio_duration for r in valid_rtf_samples)
        total_inference_time = sum(r.inference_time for r in valid_rtf_samples)
        global_rtf_x = (
            total_audio_duration / total_inference_time
            if total_inference_time > 0
            else None
        )

        print("\nDescriptive Statistics - RTFx:")
        if global_rtf_x is not None:
            print(f"  Global RTFx:    {global_rtf_x:.2f}")
        if rtf_values.size > 0:
            print(f"  Mean:           {rtf_values.mean():.2f}")
            print(f"  Median:         {np.median(rtf_values):.2f}")
            print(f"  Std deviation:  {rtf_values.std():.2f}")
            print(f"  Variance:       {rtf_values.var():.4f}")
            print(f"  1st Quartile:   {np.percentile(rtf_values, 25):.2f}")
            print(f"  3rd Quartile:   {np.percentile(rtf_values, 75):.2f}")
            print(f"  Min:            {rtf_values.min():.2f}")
            print(f"  Max:            {rtf_values.max():.2f}")
            print(f"  Count:          {len(rtf_values)}")
        else:
            print("  No valid RTFx values.")

        # Prepare dictionaries for programmatic use
        wer_stats = {
            "weighted_wer": weighted_wer,
            "mean": wer_values.mean() if wer_values.size > 0 else None,
            "median": np.median(wer_values) if wer_values.size > 0 else None,
            "std": wer_values.std() if wer_values.size > 0 else None,
            "variance": wer_values.var() if wer_values.size > 0 else None,
            "q1": np.percentile(wer_values, 25) if wer_values.size > 0 else None,
            "q3": np.percentile(wer_values, 75) if wer_values.size > 0 else None,
            "min": wer_values.min() if wer_values.size > 0 else None,
            "max": wer_values.max() if wer_values.size > 0 else None,
            "count": len(wer_values),
        }

        rtf_stats = {
            "global_rtf_x": global_rtf_x,
            "mean": rtf_values.mean() if rtf_values.size > 0 else None,
            "median": np.median(rtf_values) if rtf_values.size > 0 else None,
            "std": rtf_values.std() if rtf_values.size > 0 else None,
            "variance": rtf_values.var() if rtf_values.size > 0 else None,
            "q1": np.percentile(rtf_values, 25) if rtf_values.size > 0 else None,
            "q3": np.percentile(rtf_values, 75) if rtf_values.size > 0 else None,
            "min": rtf_values.min() if rtf_values.size > 0 else None,
            "max": rtf_values.max() if rtf_values.size > 0 else None,
            "count": len(rtf_values),
        }

        return wer_stats, rtf_stats

    def extract_errors(self, audio_path: str, reference: str, prediction: str) -> List[Dict[str, str]]:
        """
        Extract token-level errors between reference and prediction.

        Parameters
        ----------
        audio_path : str
            Path to the evaluated audio file.
        reference : str
            Ground truth transcription.
        prediction : str
            Predicted transcription.

        Returns
        -------
        List[Dict[str, str]]
            List of dictionaries with alignment errors (insertion, deletion, substitution).
        """
        reference_clean = self.normalizer.normalize(reference)
        prediction_clean = self.normalizer.normalize(prediction)

        measures = process_words(reference_clean, prediction_clean)

        errors = []
        for chunk in measures.alignments[0]:
            if chunk.type != "equal":
                ref_words = measures.references[0][chunk.ref_start_idx : chunk.ref_end_idx]
                hyp_words = measures.hypotheses[0][chunk.hyp_start_idx : chunk.hyp_end_idx]
                errors.append({
                    "path": audio_path,
                    "type_error": chunk.type,
                    "reference": " ".join(ref_words) if ref_words else "",
                    "prediction": " ".join(hyp_words) if hyp_words else "",
                })

        return errors


    def save_results(self, results: List[EvaluationResult], output_path: str):
        """
        Save the full evaluation results to a CSV file.

        Parameters
        ----------
        results : List[EvaluationResult]
            Evaluation outputs for all audio samples.
        output_path : str
            Destination file path.
        """
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved full evaluation results to {output_path}")


    def save_predictions(self, results: List[EvaluationResult], output_path: str):
        """
        Save only the audio path and transcription predictions.

        Parameters
        ----------
        results : List[EvaluationResult]
            Evaluation results from which to extract predictions.
        output_path : str
            Destination file path.
        """
        pred_data = [{"path": r.path, "prediction": r.prediction} for r in results]
        df = pd.DataFrame(pred_data)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved predictions to {output_path}")


    def save_errors(self, results: List[EvaluationResult], output_path: str):
        """
        Save detailed token-level transcription errors to CSV.

        Parameters
        ----------
        results : List[EvaluationResult]
            List of evaluation outputs to analyze.
        output_path : str
            Destination file path.
        """
        all_errors = []
        for r in results:
            errs = self.extract_errors(r.path, r.reference, r.prediction)
            all_errors.extend(errs)

        if all_errors:
            df = pd.DataFrame(all_errors)
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"Saved detailed alignment errors to {output_path}")
        else:
            print("No errors found. Nothing saved.")