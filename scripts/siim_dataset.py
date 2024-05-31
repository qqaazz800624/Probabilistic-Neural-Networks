#%%

import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from monai.transforms import Compose, LoadImaged, Resized, MapLabelValued, EnsureChannelFirstd
from monai.transforms import EnsureTyped, NormalizeIntensityd
from typing import List, Dict

class SIIMDataset(Dataset):
    """Image segmentation dataset based on a JSON data list."""

    def __init__(self, 
                 folds: List[str],
                 transform = None,
                 json_file: str = 'datalist.json',
                 data_root: str = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax',
                 if_test: bool = False
                 ):
        """
        Args:
            folds: List of fold names to use (e.g., ['training'] for training, ['validation'], ['testing'] ...).
            json_file: Filename of the JSON file containing data paths.
            transform: Optional transform to be applied on a sample.
            data_root: Root directory of the dataset.
        """
        self.json_path = os.path.join(data_root, json_file)
        with open(self.json_path) as f:
            self.data_list = json.load(f)

        self.if_test = if_test
        self.transform = transform
        self.data_root = data_root
        self.samples = []
        for fold in folds:
            self.samples.extend(self.data_list[fold])

        self.image_loader = Compose([
                            LoadImaged(keys=['image', 'target'], 
                                        reader='PILReader',
                                        ensure_channel_first=True
                                        ),
                            MapLabelValued(keys=['target'],
                                           orig_labels=[0, 255],
                                           target_labels=[0, 1]),
                            Resized(keys=['image', 'target'], 
                                    spatial_size=[512, 512],
                                    mode = ['bilinear', 'nearest']),
                                    ])
        
        self.test_transforms = Compose([
                                NormalizeIntensityd(keys=['image']),
                                EnsureTyped(keys=['image', 'target'], dtype='float32')
        ])

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_path = os.path.join(self.data_root, sample['image'])
        target_path = os.path.join(self.data_root, sample['label'])
        
        data_list = {
            'image': image_path,
            'target': target_path
        }

        data = self.image_loader(data_list)
        image = data['image']
        target = data['target']
        
        if self.transform:
            transformed = self.transform(data)
            image = transformed['image']
            target = transformed['target']

        if self.if_test:
            transformed = self.test_transforms(data)
            image = transformed['image']
            target = transformed['target']

        return {'input': image, 'target': target}

#%%





#%%