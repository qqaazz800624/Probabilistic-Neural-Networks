#%%
import os
import zipfile
import platform
import warnings
from glob import glob
from dataclasses import dataclass
 
# To filter UserWarning.
warnings.filterwarnings("ignore", category=UserWarning)
 
import cv2
import requests
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
 
# For data augmentation and preprocessing.
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
# Imports required SegFormer classes
from transformers import SegformerForSemanticSegmentation
 
# Importing lighting along with a built-in callback it provides.
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
 
# Importing torchmetrics modular and functional implementations.
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score
 
# To print model summary.
from torchinfo import summary
 
# Sets the internal precision of float32 matrix multiplications.
torch.set_float32_matmul_precision('high')
 
# To enable determinism.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import wandb
 
wandb.login()


#%%

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES:   int = 4 # including background.
    IMAGE_SIZE: tuple[int,int] = (288, 288) # W, H
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD:  tuple = (0.229, 0.224, 0.225)
    BACKGROUND_CLS_ID: int = 0
    URL: str = r"https://www.dropbox.com/scl/fi/r0685arupp33sy31qhros/dataset_UWM_GI_Tract_train_valid.zip?rlkey=w4ga9ysfiuz8vqbbywk0rdnjw&dl=1"
    DATASET_PATH: str = os.path.join(os.getcwd(), "dataset_UWM_GI_Tract_train_valid")
 
@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES: str = os.path.join(DatasetConfig.DATASET_PATH, "train", "images", r"*.png")
    DATA_TRAIN_LABELS: str = os.path.join(DatasetConfig.DATASET_PATH, "train", "masks",  r"*.png")
    DATA_VALID_IMAGES: str = os.path.join(DatasetConfig.DATASET_PATH, "valid", "images", r"*.png")
    DATA_VALID_LABELS: str = os.path.join(DatasetConfig.DATASET_PATH, "valid", "masks",  r"*.png")
         
@dataclass
class TrainingConfig:
    BATCH_SIZE:      int = 48 # 32. On colab you should be able to use batch size of 32 with T4 GPU.
    NUM_EPOCHS:      int = 100
    INIT_LR:       float = 3e-4
    NUM_WORKERS:     int = 0 if platform.system() == "Windows" else 12 # os.cpu_count()
 
    OPTIMIZER_NAME:  str = "AdamW"
    WEIGHT_DECAY:  float = 1e-4
    USE_SCHEDULER:  bool = True # Use learning rate scheduler?
    SCHEDULER:       str = "MultiStepLR" # Name of the scheduler to use.
    MODEL_NAME:      str = "nvidia/segformer-b4-finetuned-ade-512-512"
     
 
@dataclass
class InferenceConfig:
    BATCH_SIZE:  int = 10
    NUM_BATCHES: int = 2




#%%

# Create a mapping from class ID to RGB color value. Required for visualization.
id2color = {
    0: (0, 0, 0),    # background pixel
    1: (0, 0, 255),  # Stomach
    2: (0, 255, 0),  # Small Bowel
    3: (255, 0, 0),  # large Bowel
}
 
print("Number of classes", DatasetConfig.NUM_CLASSES)
 
# Reverse id2color mapping.
# Used for converting RGB mask to a single channel (grayscale) representation.
rev_id2color = {value: key for key, value in id2color.items()}


#%%







#%%









#%%







#%%