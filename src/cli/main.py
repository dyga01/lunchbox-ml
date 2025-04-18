from pathlib import Path
from typing import Optional, Literal
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
def serve(
    model: str = Option(..., "--model", "-m", help="Path to the model file"),
    backend: str = Option(..., "--backend", "-b", help="Serving backend to use"),
):
    """Serve a machine learning model locally."""
    if backend == "pytorch":
        print(f"Serving PyTorch model: {model}")
        # Add logic for PyTorch native serving
    elif backend == "onnx":
        print(f"Serving ONNX model: {model}")
        # Add logic for ONNX runtime serving
    else:
        print(f"Error: Unsupported backend '{backend}'")

if __name__ == "__main__":
    app()
