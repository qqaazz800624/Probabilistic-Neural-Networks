#!/bin/bash

# Set default path for the Python script
DEFAULT_PYTHON_SCRIPT="./scripts/pretraining_siim.py"

# Allow overriding the default path with a command-line argument
PYTHON_SCRIPT=${1:-$DEFAULT_PYTHON_SCRIPT}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=1

# Run the Python script with the specified or default path
python "$PYTHON_SCRIPT"
