#!/bin/bash

# Navigate to the optimizer directory
magic init optimizer --format pyproject
cd optimizer

# Create a Python hello world program
echo "print('Hello, World!')" > main.py

# Run the Python program
magic run python3 main.py
