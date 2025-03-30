import pytest
from typer.testing import CliRunner
from src.cli.main import app

runner = CliRunner()

def test_train_command():
    result = runner.invoke(app, ["train", "--model", "test_model"])
    assert result.exit_code == 0
    assert "Training model" in result.output

def test_test_command():
    result = runner.invoke(app, ["test", "--model", "test_model"])
    assert result.exit_code == 0
    assert "Testing model" in result.output

def test_deploy_command():
    result = runner.invoke(app, ["deploy", "--model", "test_model"])
    assert result.exit_code == 0
    assert "Deploying model" in result.output

def test_invalid_command():
    result = runner.invoke(app, ["invalid_command"])
    assert result.exit_code != 0
    assert "Error" in result.output