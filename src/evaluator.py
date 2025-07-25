from typing import Optional, List
import csv
import time
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from mistral_common.audio import Audio
from src.dataclasses import EvaluationResult
from src.normalizer import TextNormalizer
from jiwer import process_words


class VoxtralEvaluator:
    """
    Evaluator class for the Voxtral ASR model.

    This class provides methods to transcribe audio files, evaluate single samples
    against reference transcriptions by computing WER and related statistics, and
    evaluate multiple samples from a CSV annotations file.

    Attributes
    ----------
    device : str
        Device identifier for model inference ('cuda' or 'cpu').
    processor : transformers.AutoProcessor
        Tokenizer and processor compatible with the Voxtral model.
    model : transformers.VoxtralForConditionalGeneration
        Pretrained Voxtral ASR model loaded for inference.
    normalizer : TextNormalizer
        Text normalization utility to standardize transcriptions before comparison.
    """
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the evaluator by loading model and processor on the specified device.

        Parameters
        ----------
        model_path : str
            Path or identifier of the pretrained Voxtral model.
        device : Optional[str], default=None
            Device to run inference on. Defaults to CUDA if available, otherwise CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.normalizer = TextNormalizer()

    def transcribe(self, audio_path: str) -> str:
        """
        Generate a transcription from a given audio file using the Voxtral model.

        Parameters
        ----------
        audio_path : str
            Path to the audio file to transcribe.

        Returns
        -------
        str
            The predicted transcription text.
        """
        inputs = self.processor.apply_transcrition_request(
            language="fr",
            audio=audio_path,
            model_id="mistralai/Voxtral-Mini-3B-2507"
        ).to(self.device, dtype=torch.bfloat16)

        outputs = self.model.generate(**inputs, max_new_tokens=500)
        return self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

    def evaluate_sample(self, audio_path: str, reference: str) -> EvaluationResult:
        """
        Evaluate a single audio sample by computing WER and related metrics.

        The method transcribes the audio, normalizes both the prediction and the
        reference, calculates insertion, deletion, and substitution errors, measures
        inference time, and computes the inverse real-time factor (RTFx).

        Parameters
        ----------
        audio_path : str
            Path to the audio sample.
        reference : str
            Ground truth transcription text.

        Returns
        -------
        EvaluationResult
            Dataclass instance containing WER, error counts, durations, RTFx, and transcriptions.
        """
        try:
            # Load audio file and get duration (in seconds)
            audio = Audio.from_file(audio_path)
            audio_duration = audio.duration 

            # Start timing for inference
            start_time = time.perf_counter()
            prediction = self.transcribe(audio_path)
            end_time = time.perf_counter()

            # Calculate inference time and RTFx
            inference_time = end_time - start_time
            rtf_x = audio_duration / inference_time if inference_time > 0 else None

            # Normalize both reference and prediction texts
            reference_clean = self.normalizer.normalize(reference)
            prediction_clean = self.normalizer.normalize(prediction)

            # Compute detailed error statistics using JiWER
            measures = process_words(reference_clean, prediction_clean)
            insertions = measures.insertions
            deletions = measures.deletions
            substitutions = measures.substitutions
            reference_length = len(reference_clean.split())

            # Calculate WER: normalized edit distance
            sample_wer = (
                (insertions + deletions + substitutions) / reference_length
                if reference_length > 0
                else None
            )

            return EvaluationResult(
                path=audio_path,
                wer=sample_wer,
                insertions=insertions,
                deletions=deletions,
                substitutions=substitutions,
                reference_length=reference_length,
                prediction=prediction,
                reference=reference,
                rtf_x=rtf_x,
                audio_duration=audio_duration,
                inference_time=inference_time,
            )
        except Exception as e:
            print(f"Error for {audio_path} : {e}")
            return EvaluationResult(
                path=audio_path,
                wer=None,
                insertions=None,
                deletions=None,
                substitutions=None,
                reference_length=None,
                prediction="",
                reference=reference,
                rtf_x=None
            )

    def evaluate_all(self, annotations_path: str) -> List[EvaluationResult]:
        """
        Batch evaluate multiple audio samples listed in a CSV annotations file.

        The CSV file must contain at least two columns: 'path' (audio file path) and
        'reference' (ground truth transcription).

        Parameters
        ----------
        annotations_path : str
            Path to the CSV file containing evaluation annotations.

        Returns
        -------
        List[EvaluationResult]
            List of EvaluationResult objects for each sample.
        """
        results = []
        with open(annotations_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, desc="Evaluating audio files"):
                result = self.evaluate_sample(row["path"], row["reference"])
                results.append(result)
        return results