from setuptools import setup, find_packages

setup(
    name="lunchbox-ml",
    version="0.1.0",
    description="A Lightweight CLI for Local ML Model Deployment",
    author="Aidan Dyga",
    url="https://github.com/dyga01/lunchbox-ml",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "typer",
        "docker",
        "numpy",  # Add any other ML libraries you need
        "pandas",  # Example additional dependency
    ],
    entry_points={
        "console_scripts": [
            "lunchbox=cli.main:app",  # Adjust according to your Typer app
        ],
    },
)