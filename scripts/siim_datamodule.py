#%%

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from siim_dataset import SIIMDataset

from monai.transforms import Compose
from monai.transforms import EnsureTyped, NormalizeIntensityd
from augmentations import XRayAugs

class SIIMDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size_train = 32,
        num_workers_train = 4,
        batch_size_val = 16,
        num_workers_val = 4,
        batch_size_test = 8,
        num_workers_test = 2):
        """Initialize an image segmentation datamodule.

        Args:
            batch_size_train: batch size for training dataloader
            num_workers_train: number of workers for training dataloader
            batch_size_val: batch size for validation dataloader
            num_workers_val: number of workers for validation dataloader
            batch_size_test: batch size for test dataloader
            num_workers_test: number of workers for test dataloader
        """
        super().__init__()
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.num_workers_test = num_workers_test
        
        self.train_folds = ['training']
        self.val_folds = ['validation']
        self.test_folds = ['testing']

        self.train_transforms = Compose([
                                XRayAugs(img_key='image', seg_key='target'),
                                NormalizeIntensityd(keys=['image']),
                                EnsureTyped(keys=['image', 'target'], dtype='float32')
                                ])
        
        self.val_transforms = Compose([
                                NormalizeIntensityd(keys=['image']),
                                EnsureTyped(keys=['image', 'target'], dtype='float32')
                                ])
        
        self.test_transforms = Compose([
                                NormalizeIntensityd(keys=['image']),
                                EnsureTyped(keys=['image', 'target'], dtype='float32')
                                ])

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.train_folds, transform=self.train_transforms),
            batch_size=self.batch_size_train,
            num_workers=self.num_workers_train,
            drop_last=True,
            shuffle=False
        )

    def val_dataloader(self):
        """Return the val dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.val_folds, transform=self.val_transforms),
            batch_size=self.batch_size_val,
            num_workers=self.num_workers_val,
            drop_last=True,
            shuffle=False
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.test_folds, transform=self.test_transforms),
            batch_size=self.batch_size_test,
            num_workers=self.num_workers_test,
            drop_last=True,
            shuffle=False
        )

#%%




#%%





#%%