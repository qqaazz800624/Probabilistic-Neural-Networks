#!/bin/bash

# Set default path for the Python script
DEFAULT_PYTHON_SCRIPT="./scripts/prob_u_net_montgomery.py"

# Allow overriding the default path with a command-line argument
PYTHON_SCRIPT=${1:-$DEFAULT_PYTHON_SCRIPT}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=2

# Run the Python script with the specified or default path
python "$PYTHON_SCRIPT"
