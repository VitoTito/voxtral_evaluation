import sys
import typer
import time
from pathlib import Path
import logging

from src.evaluator import VoxtralEvaluator
from src.metrics import VoxtralMetrics

app = typer.Typer(add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_evaluation(
    model_path: Path,
    annotations_path: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    predictions_path = output_dir / "predictions.csv"
    errors_path = output_dir / "errors.csv"

    logging.info("Initializing evaluator and metrics...")
    evaluator = VoxtralEvaluator(str(model_path))
    metrics = VoxtralMetrics()

    logging.info("Starting evaluation...")
    start = time.perf_counter()
    results = evaluator.evaluate_all(str(annotations_path))
    elapsed = time.perf_counter() - start
    logging.info(f"Evaluation completed in {elapsed:.2f} seconds.")

    logging.info("Saving results...")
    metrics.save_results(results, str(results_path))
    metrics.save_predictions(results, str(predictions_path))
    metrics.save_errors(results, str(errors_path))
    
    logging.info("Computing statistics...")
    metrics.compute_statistics(results)
    logging.info("Done.")

@app.command()
def evaluate(
    model_path: Path = typer.Option(Path("model/"), help="Path to the Voxtral model directory"),
    annotations_path: Path = typer.Option(Path("data/annotations.csv"), help="Path to the CSV file with annotations"),
    output_dir: Path = typer.Option(Path("output/"), help="Directory where output files will be saved")
):
    """
    Evaluate the Voxtral ASR model on a dataset and generate metrics and outputs.
    """
    run_evaluation(model_path, annotations_path, output_dir)

def main():
    if len(sys.argv) > 1:
        app()
    else:
        # Call with default args directly as Path objects
        run_evaluation(
            Path("model/"),
            Path("data/annotations.csv"),
            Path("output/")
        )

if __name__ == "__main__":
    main()