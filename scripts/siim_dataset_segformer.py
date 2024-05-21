import torch
import os
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Dict
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

class SIIMDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, 
                 folds: List[str],
                 feature_extractor: SegformerFeatureExtractor,
                 transform=None,
                 data_root: str = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax', 
                 json_file: str = 'datalist.json'
                 ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.json_path = os.path.join(data_root, json_file)
        self.feature_extractor = feature_extractor
        with open(self.json_path) as f:
            self.data_list = json.load(f)

        self.transform = transform
        self.data_root = data_root

        self.samples = []
        for fold in folds:
            self.samples.extend(self.data_list[fold])

        self.classes_csv_file = os.path.join(self.root_dir, "_classes.csv")
        with open(self.classes_csv_file, 'r') as fid:
            data = [l.split(',') for i,l in enumerate(fid) if i !=0]
        self.id2label = {x[0]:x[1] for x in data}
        
        image_file_names = [f for f in os.listdir(self.root_dir) if '.jpg' in f]
        mask_file_names = [f for f in os.listdir(self.root_dir) if '.png' in f]
        
        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.root_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.root_dir, self.masks[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs