# voxtral_eval
<p align="center">
  <img src="data/voxtral.png" width="600" alt="Voxtral (Mistral AI)">
</p>

A reproducible evaluation pipeline for the **Voxtral Mini 1.0 (3B)** ASR model, enabling inference and benchmarking with key metrics (Word Error Rate and Character Error Rate).

<p align="center">
  <img src="data/voxtral.png" width="600" alt="Voxtral (Mistral AI)">
</p>


## Objective

This repository provides a reproducible pipeline to **evaluate Voxtral models**, starting with [Voxtral Mini 1.0 (3B)](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), released on **July 15th, 2025** by [Mistral AI](https://mistral.ai/news/voxtral). You can : 

- Run inference on audio datasets
- Compute standard ASR performance metrics: **Word Error Rate (WER)** and **Character Error Rate (CER)**. For more information about these metrics, check [WER](https://huggingface.co/spaces/evaluate-metric/wer)
- Save results in CSV for further analysis


## Structure

- `.github/workflows` : contains workflows for tests
- `data/` : contains audio datasets and annotations
- `results/` : contains evaluation results (CSV files)
- `src/` : contains source code
- `tests/` : contains unit tests
- `main.py` : contains main entry point for evaluation
- `README.md` : contains documentation
- `requirements.txt` : contains Python dependencies


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
source voxtral_env/bin/activate   # Linux/macOS
voxtral_env\Scripts\activate     # Windows
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

4. **Prepare your data**

- Place your audio files in the `data/user_audio/audio_files/` folder (or elsewhere, but update your CSV accordingly).
- Prepare a CSV file `data/user_audio/annotations/annotations.csv` with at least two columns:
     - `path`: relative or absolute paths to the audio files (e.g. `data/audio_files/example.wav`)
     - `reference`: the corresponding ground truth transcription.

If no user audio is found, the pipeline will automatically use the default test dataset `data/test_samples/`

5. **Run the evaluation**

You can run with default paths:

```bash
python main.py
```

6. **Check output!**

- Evaluation results are saved as CSV files in the `results/` folder
- Columns include: `path`, `reference`, `prediction`, `wer`, `cer`


## Tests

Unit tests are implemented using the standard `unittest` framework.

Run tests locally with:

```bash
python -m unittest discover -s Tests
```

## Coming in the Future

- Benchmarking results on public datasets
- Visualization

Stay tuned!
