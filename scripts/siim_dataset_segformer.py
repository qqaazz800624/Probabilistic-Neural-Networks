#%%
import torch
import os
import json
from torch.utils.data import Dataset
from torchvision.transforms.transforms import RandomRotation, ColorJitter, RandomResizedCrop
from typing import List, Dict

import albumentations as A
from monai.transforms import AsDiscrete, RandAffine
from albumentations.pytorch import ToTensorV2
import cv2
from dataclasses import dataclass
import platform

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES:   int = 2 # including background.
    IMAGE_SIZE: tuple[int,int] = (512, 512) # W, H
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD:  tuple = (0.229, 0.224, 0.225)
    CHANNELS: int = 3
    DATASET_PATH: str = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'

@dataclass
class TrainingConfig:
    BATCH_SIZE_TRAIN:int = 32 # 32. On colab you should be able to use batch size of 32 with T4 GPU.
    NUM_EPOCHS:      int = 100
    INIT_LR:       float = 3e-4
    NUM_WORKERS_TRAIN:int = 0 if platform.system() == "Windows" else 8 # os.cpu_count()
 
    OPTIMIZER_NAME:  str = "AdamW"
    WEIGHT_DECAY:  float = 1e-4
    USE_SCHEDULER:  bool = True # Use learning rate scheduler?
    SCHEDULER:       str = "MultiStepLR" # Name of the scheduler to use.
    #MODEL_NAME:      str = "nvidia/segformer-b0-finetuned-ade-512-512"  # pretrained model name.
    MODEL_NAME:      str = "nvidia/segformer-b4-finetuned-ade-512-512"  # pretrained model name.

@dataclass
class InferenceConfig:
    BATCH_SIZE:  int = 10
    NUM_BATCHES: int = 2


#%%

class SIIMDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, 
                 folds: List[str],
                 ds_mean: List[float] = DatasetConfig.MEAN,
                 ds_std: List[float] = DatasetConfig.STD,
                 img_size: tuple[int,int] = DatasetConfig.IMAGE_SIZE,
                 data_root: str = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax', 
                 json_file: str = 'datalist.json',
                 is_train: bool = False
                 ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.json_path = os.path.join(data_root, json_file)
        self.img_size = img_size
        
        with open(self.json_path) as f:
            self.data_list = json.load(f)

        self.data_root = data_root

        self.samples = []
        for fold in folds:
            self.samples.extend(self.data_list[fold])

        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.is_train = is_train
        self.transforms  = self.setup_transforms(mean=self.ds_mean, std=self.ds_std)
        self.discreter = AsDiscrete(threshold=0.5)

    def __len__(self):
        return len(self.samples)
    
    def setup_transforms(self, *, mean, std):
        transforms = []
 
        # Augmentation to be applied to the training set.
        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5), 
                A.VerticalFlip(p=0.5),
                RandomRotation(degrees=15, p=0.5),
                # RandAffine(prob=0.5,
                #             rotate_range=0.25,
                #             shear_range=0.2,
                #             translate_range=0.1,
                #             scale_range=0.2,
                #             padding_mode='zeros')
                # ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                # RandomResizedCrop(224, scale=(0.8, 1.0), p=0.5),
                # A.ShiftScaleRotate(scale_limit=0.12, rotate_limit=0.15, shift_limit=0.12, p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                # A.CoarseDropout(max_holes=8, max_height=self.img_size[1]//20, max_width=self.img_size[0]//20, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5)
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
        
        sample = self.samples[index]
        image_path = os.path.join(self.data_root, sample['image'])
        mask_path = os.path.join(self.data_root, sample['label'])
        image = self.load_file(image_path, depth=cv2.IMREAD_COLOR)
        mask = self.load_file(mask_path, depth=cv2.IMREAD_GRAYSCALE)

        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        #mask = self.discreter(mask.unsqueeze(0))
        mask = self.discreter(mask).long()
        return image, mask
    
#%%

import cv2

cv2.IMREAD_GRAYSCALE


#%%