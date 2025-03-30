#!/bin/sh

# Entry point for the Lunchbox ML Docker container

# Set up any necessary environment variables
export MODEL_DIR=/app/models
export CONFIG_FILE=/app/config.yaml

# Run the main application
python -m src.cli.main "$@"