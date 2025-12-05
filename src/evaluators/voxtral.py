"""
VoxtralEvaluator module.

Provides a wrapper class around the Voxtral ASR model to transcribe audio files.
"""

from typing import Optional
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor


class VoxtralEvaluator:
    """
    Wrapper for Voxtral ASR model.

    Methods
    -------
    transcribe(audio_path: str) -> str
        Transcribes the given audio file and returns the predicted text.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the Voxtral model.

        Parameters
        ----------
        model_path : str
            Path to the Voxtral model (local or Hugging Face identifier).
        device : str, optional
            Device to run the model on ('cuda' or 'cpu'). Defaults to auto-detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file using Voxtral.

        Parameters
        ----------
        audio_path : str
            Path to the audio file to transcribe.

        Returns
        -------
        str
            Predicted transcription.
        """
        inputs = self.processor.apply_transcription_request(
            language="fr",
            audio=audio_path,
            model_id="mistralai/Voxtral-Mini-3B-2507"
        ).to(self.device, dtype=torch.bfloat16)

        outputs = self.model.generate(**inputs, max_new_tokens=500)
        return self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]