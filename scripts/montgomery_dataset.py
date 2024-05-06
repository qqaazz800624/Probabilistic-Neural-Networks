#%%

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict

class MontgomeryDataset(Dataset):
    """Image segmentation dataset based on a JSON data list."""

    def __init__(self, 
                 folds: List[str],
                 transform=None,
                 json_path: str = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet/datalist_fold_montgomery.json',
                 data_root: str = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet'
                 ):
        """
        Args:
            folds: List of fold names to use (e.g., ['fold_0', 'fold_1', 'fold_2'] for training).
            json_path: Path to the JSON file containing data paths.
            transform: Optional transform to be applied on a sample.
            data_root: Root directory of the dataset.
        """
        with open(json_path) as f:
            self.data_list = json.load(f)

        self.transform = transform
        self.data_root = data_root
        self.samples = []
        for fold in folds:
            self.samples.extend(self.data_list[fold])

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_path = os.path.join(self.data_root, sample['image'])
        target_path = os.path.join(self.data_root, sample['target'])
        
        data_list = {
            'image': image_path,
            'target': target_path
        }

        if self.transform:
            transformed = self.transform(data_list)
            image = transformed['image'].unsqueeze(0)
            target = transformed['target'].unsqueeze(0)

        return {'image': image, 'target': target}


#%%


'''

import json 
from monai.transforms import Compose,LoadImaged, Resized, ScaleIntensityd
from monai.transforms import ConcatItemsd, RandAffined, DeleteItemsd, EnsureTyped
from manafaln.transforms import LoadJSONd, ParseXAnnotationSegmentationLabeld, Interpolated, Filld, OverlayMaskd

train_transforms = Compose([
                        LoadImaged(keys=['image', 'target']),
                                ])


data_root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet'
datalist_path = os.path.join(data_root, 'datalist_fold_montgomery.json')

with open(datalist_path) as f:
     data_list = json.load(f)

sample = data_list['fold_0'][0]

image_path = os.path.join(data_root, sample['image'])
target_path = os.path.join(data_root, sample['target'])

data_dict = {
    'image': image_path, 
    'target': target_path
}

transformed = train_transforms(data_dict)
transformed['image'].shape, transformed['target'].shape
transformed['target']

'''




#%%



