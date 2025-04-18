from pathlib import Path
from typing import Optional
from typer import Typer, Option
from src.train.python_train import run_model

app = Typer()

@app.command()
def train(
    model: str = Option(..., "--model", "-m", help="Path to the model file"),
    output: bool = Option(False, "--output", "-o", help="Report model output"),
    benchmark: bool = Option(False, "--benchmark", "-b", help="Report performance benchmarks"),
):
    """Train a machine learning model."""
    model = Path(model)
    if model.is_file():
        # Run the model and print the models results based on the command
        run_model(model, output, benchmark)
    else:
        print(f"Error: Model file {model} does not exist.")

@app.command()
def deploy(model_name: str = Option(..., "--model", "-m", help="Name of the model to deploy")):
    """Deploy a machine learning model using Docker."""
    print(f"Deploying model: {model_name}")

if __name__ == "__main__":
    app()
