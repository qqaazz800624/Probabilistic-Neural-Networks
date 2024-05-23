#!/bin/bash

# Set default path for the Python script
DEFAULT_PYTHON_SCRIPT="./scripts/pretraining_siim.py"

# Allow overriding the default path with a command-line argument
PYTHON_SCRIPT=${1:-$DEFAULT_PYTHON_SCRIPT}

# Set default CUDA device to use
DEFAULT_CUDA_DEVICE="1"

# Allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${2:-$DEFAULT_CUDA_DEVICE}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Run the Python script with the specified or default path
python "$PYTHON_SCRIPT"
