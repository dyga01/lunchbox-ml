from pathlib import Path
from typer import Typer, Option
from src.train.train import run_model

app = Typer()

@app.command()
def train(model: str = Option(..., "--model", "-m", help="Path to the model file")):
    """Train a machine learning model."""
    model = Path(model)
    if model.is_file():
        print(f"\nTraining model from file: {model}\n")
        run_model(model)
        # Add logic to load and train the model from the file
    else:
        print(f"Error: Model file {model} does not exist.")

@app.command()
def deploy(model_name: str = Option(..., "--model", "-m", help="Name of the model to deploy")):
    """Deploy a machine learning model using Docker."""
    print(f"Deploying model: {model_name}")

if __name__ == "__main__":
    app()
