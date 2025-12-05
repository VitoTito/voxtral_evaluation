import os
import csv
from src.core.evaluation import EvaluatorPipeline

# ------------------------
# Configuration
# ------------------------
MODEL_PATH = "mistralai/Voxtral-Mini-3B-2507"  # path to the Voxtral model (local or Hugging Face)
USER_AUDIO_DIR = "data/user_audio/audio_files"  # folder for user audio files
TEST_SAMPLES_DIR = "data/test_samples/audio_files"  # folder for test sample audio files
OUTPUT_DIR = "results"  # folder where CSV results will be saved


def save_results_csv(results, output_path: str):
    """
    Save evaluation results to a CSV file.

    Parameters
    ----------
    results : list of EvaluationResult
        List of evaluation results returned by the EvaluatorPipeline.
    output_path : str
        Full path of the CSV file to create.
    """
    fieldnames = ["path", "reference", "prediction", "wer", "cer"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "path": r.path,
                "reference": r.reference,
                "prediction": r.prediction,
                "wer": r.wer,
                "cer": r.cer
            })


def get_dataset_to_evaluate():
    """
    Determine which dataset to use for evaluation.

    If the `user_audio` folder contains audio files, it will be used.
    Otherwise, it falls back to the reference test dataset `test_samples`.

    Returns
    -------
    dataset_name : str
        Name of the chosen dataset ("user_audio" or "test_samples").
    annotations_path : str
        Path to the corresponding annotations CSV file.
    """
    if os.path.exists(USER_AUDIO_DIR) and os.listdir(USER_AUDIO_DIR):
        print("User audio detected. Using user_audio dataset.")
        annotations_path = "data/user_audio/annotations/annotations.csv"
        dataset_name = "user_audio"
    else:
        print("No user audio detected. Falling back to test_samples dataset.")
        annotations_path = "data/test_samples/annotations/annotations.csv"
        dataset_name = "test_samples"
    return dataset_name, annotations_path


def main():
    """
    Main entry point.

    - Creates the results folder if it doesn't exist
    - Selects which dataset to evaluate
    - Runs the evaluation pipeline
    - Saves the results to a CSV file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine dataset
    dataset_name, annotations_path = get_dataset_to_evaluate()

    # Initialize evaluation pipeline
    pipeline = EvaluatorPipeline(model_path=MODEL_PATH)
    print(f"Evaluating dataset: {dataset_name}")

    # Run evaluation
    results = pipeline.evaluate_all(annotations_path)

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_results.csv")
    save_results_csv(results, output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()