# voxtral_eval
A reproducible evaluation pipeline for the **Voxtral Mini 1.0 (3B)** ASR model, enabling inference and benchmarking with key metrics such as WER and latency.


## Objective

This repository provides a reproducible pipeline to **evaluate Voxtral models**, starting with [Voxtral Mini 1.0 (3B)](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), released on **July 15th, 2025** by [Mistral AI](https://mistral.ai/news/voxtral). You can : 

- Running inference on audio datasets
- Computing standard ASR performance metrics: **Word Error Rate (WER)** and latency measures (RTFx). For more information about these metrics, check [WER](https://huggingface.co/spaces/evaluate-metric/wer) or [RTFx](https://github.com/NVIDIA/DeepLearningExamples/blob/master/Kaldi/SpeechRecognition/README.md#metrics)


## Model Used

We evaluate the model available on Hugging Face:
[`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)


## Prerequisites

- Python 3.8 or higher  
- CUDA-enabled GPU (optional but highly recommended for performance)  
- Virtual environment (recommended)  


## Setup & Usage Guide

1. **Clone the repository:**

```bash
git clone https://github.com/VitoTito/voxtral_eval.git
cd voxtral_eval
```

2. **Create and activate a Python virtual environment:**

```bash
python3 -m venv voxtral_env
source voxtral_env/bin/activate
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

4. **Prepare your data**

- Place your audio files in the `data/audio_files/` folder (or elsewhere, but update your CSV accordingly).
- Prepare a CSV file `data/annotations.csv` with at least two columns:
     - `path`: relative or absolute paths to the audio files (e.g. `data/audio_files/example.wav`)
     - `reference`: the corresponding ground truth transcription.

5. **Run the evaluation**

You can run with default paths:

```bash
python main.py
```

Or you can use the CLI options : 

```bash
python main.py --model-path model/ --annotations-path data/annotations.csv --output-dir output/
```

6. **Check output!**

- Results CSV: output/results.csv
- Predictions CSV: output/predictions.csv
- Errors CSV: output/errors.csv


## Coming Soon

- Benchmarking results on public datasets
- CER (Character Error Rate)

Stay tuned!
