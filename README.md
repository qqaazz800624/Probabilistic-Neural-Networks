# Probabilistic-Neural-Networks
Probabilistic Neural Networks for Segmentation Models with PyTorch, Lightning and Lightning UQ Box


![License](https://img.shields.io/badge/license-Apache%202-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.7%2B-orange.svg)

This repository contains code for training and evaluating Probabilistic Neural Networks (PNNs) for various machine learning tasks. The code is built using PyTorch and PyTorch Lightning for streamlined training and testing.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Probabilistic Neural Networks (PNNs) incorporate probabilistic layers and loss functions to provide uncertainty estimates along with predictions. This can be particularly useful in fields where understanding model confidence is crucial, such as medical imaging and autonomous driving.

## Features

- Implementation of Probabilistic U-Net U-Net models
- Custom loss functions including BCEWithLogitsLoss and MaskedBCEWithLogitsLoss
- Support for multi-GPU training with PyTorch Lightning
- Data augmentation using Albumentations and MONAI
- Experiment tracking and logging with WandB

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/qqaazz800624/Probabilistic-Neural-Networks.git
   cd Probabilistic-Neural-Networks
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Prepare your datasets and update the configuration files accordingly. Ensure that your data is stored in the correct format as required by the dataset classes.


## Models

The following models are implemented in this repository:
- Probabilistic U-Net
- Probabilistic U-Net (Self-Correction Based on Uncertainty)

### Custom Loss Functions

- BCEWithLogitsLoss
- MaskedBCEWithLogitsLoss


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The Probabilistic U-Net implementation is inspired by the original paper by Kohl et al.
- Special thanks to the PyTorch and PyTorch Lightning communities for their excellent libraries.

