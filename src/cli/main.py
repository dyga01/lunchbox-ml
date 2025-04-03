from pathlib import Path
from typing import Optional
from typer import Typer, Option
from src.train.train import run_model, print_benchmark_results

app = Typer()

@app.command()
def train(
    model: str = Option(..., "--model", "-m", help="Path to the model file"),
    output: bool = Option(False, "--output", "-o", help="Report model output"),
    benchmark: bool = Option(False, "--benchmark", "-b", help="Report performance benchmarks"),
    optimize: str = Option("none", "--optimize", "-opt", help="Optimization technique to use (none, cpu, gpu, memory, mixed)"),
):
    """Train a machine learning model."""
    model = Path(model)
    if model.is_file():
        # Run the model and print the models results based on the command
        metrics = run_model(model, output, benchmark, optimize)
        print_benchmark_results(model.name, metrics, show_output=output, show_error=True)
    else:
        print(f"Error: Model file {model} does not exist.")

@app.command()
def deploy(model_name: str = Option(..., "--model", "-m", help="Name of the model to deploy")):
    """Deploy a machine learning model using Docker."""
    print(f"Deploying model: {model_name}")

if __name__ == "__main__":
    app()
