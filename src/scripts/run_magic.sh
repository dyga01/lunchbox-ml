#!/bin/bash

# Navigate to the optimizer directory
magic init optimizer --format pyproject
cd optimizer

# copy all of hte files in test_models to here
cp ../test_models/* .
magic add pytorch
magic add matplotlib
magic add numpy
magic add pandas
magic add scikit-learn

# Run the Python program
magic run python3 gru_model.py
