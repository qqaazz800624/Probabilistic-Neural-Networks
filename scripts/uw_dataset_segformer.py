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
#import matplotlib.pyplot as plt
 
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
# For data augmentation and preprocessing.
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
# Imports required SegFormer classes
from transformers import SegformerForSemanticSegmentation
   
# Sets the internal precision of float32 matrix multiplications.
torch.set_float32_matmul_precision('high')
 
# To enable determinism.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import wandb
 
wandb.login()

# API key: 6cd2fb09d761b5366827b473091f79fcc7e8f959

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
    CHANNELS: int = 3
 
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

class UW_SegformerDataset(Dataset):
    def __init__(self, *, image_paths, mask_paths, img_size, ds_mean, ds_std, is_train=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths  
        self.is_train    = is_train
        self.img_size    = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.transforms  = self.setup_transforms(mean=self.ds_mean, std=self.ds_std)
 
    def __len__(self):
        return len(self.image_paths)
 
    def setup_transforms(self, *, mean, std):
        transforms = []
 
        # Augmentation to be applied to the training set.
        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.12, rotate_limit=0.15, shift_limit=0.12, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=self.img_size[1]//20, max_width=self.img_size[0]//20, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5)
            ])
 
        # Preprocess transforms - Normalization and converting to PyTorch tensor format (HWC --> CHW).
        transforms.extend([
                A.Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(always_apply=True),  # (H, W, C) --> (C, H, W)
        ])
        return A.Compose(transforms)
 
    def load_file(self, file_path, depth=0):
        file = cv2.imread(file_path, depth)
        if depth == cv2.IMREAD_COLOR:
            file = file[:, :, ::-1]
        return cv2.resize(file, (self.img_size), interpolation=cv2.INTER_NEAREST)
 
    def __getitem__(self, index):
        # Load image and mask file.
        image = self.load_file(self.image_paths[index], depth=cv2.IMREAD_COLOR)
        mask  = self.load_file(self.mask_paths[index],  depth=cv2.IMREAD_GRAYSCALE)
         
        # Apply Preprocessing (+ Augmentations) transformations to image-mask pair
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"].to(torch.long)
        return image, mask
    
#%%