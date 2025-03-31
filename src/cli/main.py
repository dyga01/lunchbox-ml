from typer import Typer, Option

app = Typer()

@app.command()
def train(model_name: str = Option(..., "--model", "-m", help="Name of the model to train")):
    """Train a machine learning model."""
    print(f"Training model: {model_name}")

@app.command()
def test(model_name: str = Option(..., "--model", "-m", help="Name of the model to test")):
    """Test a machine learning model."""
    print(f"Testing model: {model_name}")

@app.command()
def deploy(model_name: str = Option(..., "--model", "-m", help="Name of the model to deploy")):
    """Deploy a machine learning model using Docker."""
    print(f"Deploying model: {model_name}")

if __name__ == "__main__":
    app()
