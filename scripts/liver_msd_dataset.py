#%%

import json
import os
from torch.utils.data import Dataset
import torch
import nibabel as nib
from typing import Dict



class MSDLiverDataset(Dataset):
    """CT segmentation dataset for liver based on the MSD dataset."""

    def __init__(self, 
                 phase: str,  # 'training' or 'test'
                 transform=None, 
                 json_path: str = '/Task03_Liver/tumor_MSD_datalist.json',
                 data_root: str = '/data2/open_dataset/MSD'):
        """Initialize the dataset.

        Args:
            json_path: Path to the JSON file containing data paths.
            phase: 'training' or 'test' to specify which data to load.
            transform: Optional transform to be applied on a sample.
            data_root: Root directory of the dataset.
        """
        with open(json_path) as f:
            self.data_info = json.load(f)

        self.transform = transform
        self.data_root = data_root
        self.samples = self.data_info[phase]

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Construct the full path for image and label
        image_path = os.path.join(self.data_root, sample['image'])
        label_path = os.path.join(self.data_root, sample['label']) if 'label' in sample else None

        # Load the image and label using nibabel
        image = nib.load(image_path).get_fdata(dtype='float32')
        label = nib.load(label_path).get_fdata(dtype='float32') if label_path else None

        # Convert numpy arrays to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label).unsqueeze(0) if label else None

        data_dict = {
            'input': image,
            'target': label
        }

        # Apply transforms if any
        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

# Example usage:
# For training
train_dataset = MSDLiverDataset(phase='training', json_path='/mnt/data/dataset.json', data_root='/path/to/msd/root/')
# For testing
test_dataset = MSDLiverDataset(phase='test', json_path='/mnt/data/dataset.json', data_root='/path/to/msd/root/')



#%%