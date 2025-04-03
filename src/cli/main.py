from pathlib import Path
from typing import Optional
from typer import Typer, Option
from src.train.train import run_model, print_benchmark_results

app = Typer()

@app.command()
def train(
    model: str = Option(..., "--model", "-m", help="Path to the model file"),
    report_output: bool = Option(False, "--report-output", "-o", help="Report model output"),
    benchmark: bool = Option(False, "--benchmark", "-b", help="Report performance benchmarks"),
):
    """Train a machine learning model."""
    model = Path(model)
    if model.is_file():
        print(f"\nTraining model from file: {model}\n")
        # Run the model and print the models results based on the command
        metrics = run_model(model, report_output, benchmark)
        print_benchmark_results(model.name, metrics, show_output=report_output, show_error=True)
    else:
        print(f"Error: Model file {model} does not exist.")

@app.command()
def deploy(model_name: str = Option(..., "--model", "-m", help="Name of the model to deploy")):
    """Deploy a machine learning model using Docker."""
    print(f"Deploying model: {model_name}")

if __name__ == "__main__":
    app()
