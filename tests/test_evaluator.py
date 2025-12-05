"""
Tests for the Voxtral evaluation pipeline.

Mocks the VoxtralEvaluator to avoid loading the heavy model.
Checks that WER and CER computations work as expected on a mocked transcription.
"""

import unittest
from unittest.mock import patch, MagicMock
from src.preprocessing.normalizer import TextNormalizer
from src.metrics.wer import compute_wer
from src.metrics.cer import compute_cer
from src.evaluators.voxtral import VoxtralEvaluator


class TestEvaluatorPipeline(unittest.TestCase):
    """
    Test the evaluation pipeline without actually loading the heavy Voxtral model.
    Fully mocks VoxtralEvaluator to avoid network calls or GPU/CPU loading.
    """

    def setUp(self):
        self.normalizer = TextNormalizer()
        self.audio_path = "data/test_samples/audio_files/test_mono.wav"
        self.reference_text = (
            "et voilà nous sommes pris dans un embouteillage à cette heure-ci "
            "c'est bizarre il doit y avoir une manifestation quelque part "
            "quelle heure est-il il est cinq heures moins le quart "
            "la séance commence à quelle heure déjà à cinq heures pile "
            "et le film quinze minutes après et david il doit arriver à quelle heure "
            "à moins vingt on a rendez-vous à l'entrée du cinéma "
            "on est en retard essaye de l'appeler sur son portable "
            "oh mais regarde c'est david lui aussi il est pris dans l'embouteillage"
        )

    @patch.object(VoxtralEvaluator, "__init__", lambda self, model_path, device=None: None)
    @patch.object(VoxtralEvaluator, "transcribe")
    def test_pipeline_with_mocked_transcription(self, mock_transcribe):
        # Simule la transcription
        mock_transcribe.return_value = (
            "et voilà nous sommes pris dans un embouteillage à cette heure-ci "
            "c'est bizarre il doit y avoir une manifestation quelque part "
            "quelle heure est-il il est cinq heures moins le quart "
            "la séance commence à quelle heure déjà à cinq heures pile "
            "et le film quinze minutes après et david il doit arriver à quelle heure "
            "à moins vingt on a rendez-vous à l'entrée du cinéma "
            "on est en retard essaye de l'appeler sur son portable "
            "oh mais regarde c'est david lui aussi il est pris dans l'embouteillage"
        )

        # Crée l'objet sans init réel
        evaluator = VoxtralEvaluator(model_path="fake/path")
        
        # Appelle la méthode mockée
        predicted_text = evaluator.transcribe(self.audio_path)

        # Normalisation
        reference_norm = self.normalizer.normalize(self.reference_text)
        predicted_norm = self.normalizer.normalize(predicted_text)

        # Calcul métriques
        wer = compute_wer(reference_norm, predicted_norm)
        cer = compute_cer(reference_norm, predicted_norm)

        # Assertions simples
        self.assertAlmostEqual(wer, 0.0)
        self.assertAlmostEqual(cer, 0.0)


if __name__ == "__main__":
    unittest.main()