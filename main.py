import os
import csv
from src.core.evaluation import EvaluatorPipeline

# Config
MODEL_PATH = "mistralai/Voxtral-Mini-3B-2507"  # ou chemin local
USER_AUDIO_DIR = "data/user_audio/audio_files"
TEST_SAMPLES_DIR = "data/test_samples/audio_files"
OUTPUT_DIR = "results"


def save_results_csv(results, output_path):
    """Save evaluation results as CSV"""
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
    """Decide which dataset to evaluate based on user_audio folder content"""
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_name, annotations_path = get_dataset_to_evaluate()

    pipeline = EvaluatorPipeline(model_path=MODEL_PATH)
    print(f"Evaluating dataset: {dataset_name}")

    results = pipeline.evaluate_all(annotations_path)

    output_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_results.csv")
    save_results_csv(results, output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()