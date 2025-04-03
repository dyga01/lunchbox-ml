from setuptools import setup, find_packages

setup(
    name="lunchbox-ml",
    version="0.1.0",
    description="A Lightweight CLI for Local ML Model Deployment",
    author="Aidan Dyga",
    url="https://github.com/dyga01/lunchbox-ml",
    packages=find_packages(),
    install_requires=[
        "typer",
        "docker",
        "numpy",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "lunchbox=src.cli.main:app",
        ],
    },
)