# Probabilistic-Neural-Networks
This repository contains code for training and evaluating Probabilistic Neural Networks (PNNs) for pneumothorax segmentation using the SIIM-ACR Pneumothorax dataset. The code is built using PyTorch and PyTorch Lightning for streamlined training and testing. This project aims to leverage uncertainty produced by Probabilistic U-Net to develop an adaptive self-correction procedure, guiding the segmentation model to improve in areas where the model is uncertain.
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
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Probabilistic Neural Networks (PNNs) incorporate probabilistic layers and loss functions to provide uncertainty estimates along with predictions. This is particularly useful in medical imaging, where understanding model confidence is crucial. In this project, we aim to leverage the uncertainty produced by Probabilistic U-Net to guide the segmentation model to explore and improve in areas where the model finds itself uncertain.

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

### Adaptive Self-Correction Procedure
This procedure leverages the uncertainty produced by the Probabilistic U-Net to guide the model:

- Perform initial segmentation and uncertainty estimation.
- Identify uncertain regions using the uncertainty heatmap.
- Guide the model to focus on these regions for further training.

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
- The SIIM-ACR Pneumothorax dataset was used for model training and evaluation.

