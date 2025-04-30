"""CLI for training and serving machine learning models."""

from pathlib import Path
from typing import Optional, Literal
from typer import Typer, Option
from src.train.python_train import run_model
from src.deploy.parse_yml import parse_yaml_file
from src.deploy.build_deployment import build_deployment_script

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
    config: str = Option(..., "--config", "-c", help="Path to the config file"),
    benchmark: bool = Option(False, "--benchmark", "-b", help="Report serving benchmarks"),
):
    """Serve a machine learning model locally."""
    config_path = Path(config)
    if config_path.is_file():
        # Generate the deployment script
        build_deployment_script(config_path, benchmark)
    else:
        print(f"Error: Config file {config} does not exist.")

if __name__ == "__main__":
    """Entry point for the CLI application. Initializes the Typer app and listens for commands."""
    app()
