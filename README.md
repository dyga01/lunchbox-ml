# Lunchbox ML: A Lightweight CLI for Local ML Model Deployment

**NOTE**: This project is in the early stages of development and is not yet fully functional.

## Installation

To get started with Lunchbox ML, clone the repository and install the required dependencies:

```bash
git clone https://github.com/dyga01/lunchbox-ml.git
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training Models

```bash
lunchbox --help
lunchbox train --model ./test_model/main.py
lunchbox train --model ./test_model/main.py --output
lunchbox train --model ./test_model/main.py --output --benchmark
```

### Deploying Models

```bash
lunchbox serve --config ./test_model/config.yml
lunchbox serve --config ./test_model/config.yml --benchmark
```

## Overview

Lunchbox ML is a lightweight command-line interface (CLI) tool designed to simplify the deployment of machine learning models locally. It provides a streamlined workflow for training models and serving predictions, all within a local environment. The tool is particularly useful for developers who want to test and iterate on smaller ML models without relying on cloud-based solutions.

## Motivation

The motivation behind Lunchbox ML is to address the challenges of deploying machine learning models in local environments. Many existing tools focus on cloud-based deployment, which can be overkill for small-scale projects or prototyping. Lunchbox ML bridges this gap by offering a simple, efficient, and portable solution for local ML model deployment.

## Ideal Outcome

The ideal outcome of using Lunchbox ML is to enable users to:

- Train and benchmark local machine learning models with minimal configuration.
- Automatically generate code that can be used for basic deployment.
