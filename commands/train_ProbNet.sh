#!/bin/bash

# Set default path for the Python script
DEFAULT_PYTHON_SCRIPT="./scripts/prob_u_net_siim.py"

# Allow overriding the default path with a command-line argument
PYTHON_SCRIPT=${2:-$DEFAULT_PYTHON_SCRIPT}

# Set default CUDA device to use
DEFAULT_CUDA_DEVICE="1"

# Allow overriding the default CUDA device with a command-line argument
CUDA_DEVICE=${1:-$DEFAULT_CUDA_DEVICE}

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Set PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python script with the specified or default path
python "$PYTHON_SCRIPT"
