#%%

import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Dict

class SIIMDataset(Dataset):
    """Image segmentation dataset based on a JSON data list."""

    def __init__(self, 
                 folds: List[str],
                 transform=None,
                 json_file: str = 'datalist.json',
                 data_root: str = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'
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
        target_path = os.path.join(self.data_root, sample['label'])
        
        data_list = {
            'image': image_path,
            'target': target_path
        }

        if self.transform:
            transformed = self.transform(data_list)
            image = transformed['image']
            target = transformed['target']

        return {'input': image, 'target': target}


#%%


# import json 
# from monai.transforms import Compose,LoadImaged, Resized, ScaleIntensityd
# from monai.transforms import EnsureTyped, AsDiscreted
# from manafaln.transforms import LoadJSONd, ParseXAnnotationSegmentationLabeld, Interpolated, Filld, OverlayMaskd

# train_transforms = Compose([
#                         LoadImaged(keys=['image', 'target'], ensure_channel_first=True),
#                         Resized(keys=['image', 'target'], 
#                                         spatial_size=[512, 512]),
#                         ScaleIntensityd(keys=['image', 'target']),
#                         AsDiscreted(keys=['target'], threshold=0.5),
#                         EnsureTyped(keys=['image', 'target'], dtype='float32')
#                                 ])


# data_root = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'
# datalist_path = os.path.join(data_root, 'datalist.json')

# with open(datalist_path) as f:
#      data_list = json.load(f)


# sample = data_list['training'][0]

# image_path = os.path.join(data_root, sample['image'])
# target_path = os.path.join(data_root, sample['label'])


# data_dict = {
#     'image': image_path, 
#     'target': target_path
# }

# transformed = train_transforms(data_dict)

# # #%%

# transformed

# # #%%
# transformed['image'].shape, transformed['target'].shape


# # #%%

# import matplotlib.pyplot as plt

# plt.imshow(transformed['image'][0].T, cmap='gray')


# # #%%


# import matplotlib.pyplot as plt

# plt.imshow(transformed['target'][0].T, cmap='gray')


# # #%%
# from monai.transforms import AsDiscrete
# import torch

# # discreter = AsDiscrete(threshold=0.5)
# # torch.unique(discreter(transformed['target'][0]))

# torch.unique(transformed['target'][0].T)


# #%%

# import matplotlib.pyplot as plt
# from monai.transforms import AsDiscrete

# discreter = AsDiscrete(threshold=0.5)
# plt.imshow(discreter(transformed['target'][0]).T, cmap='gray')


#%%