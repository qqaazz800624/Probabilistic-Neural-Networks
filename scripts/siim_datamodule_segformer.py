#%%

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from siim_dataset_segformer import SIIMDataset


class SIIMDataModuleSegFormer(LightningDataModule):
    def __init__(
        self,
        num_classes: int = 2,
        img_size: tuple[int,int] = (384, 384),
        ds_mean = (0.485, 0.456, 0.406),
        ds_std=(0.229, 0.224, 0.225),
        batch_size_train = 32,
        num_workers_train = 8,
        batch_size_val = 16,
        num_workers_val = 4,
        batch_size_test = 8,
        num_workers_test = 2,
        pin_memory=False,
        shuffle_validation=False):
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
        self.num_classes = num_classes
        self.img_size = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.pin_memory = pin_memory
        self.shuffle_validation = shuffle_validation

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.num_workers_test = num_workers_test
        
        self.train_folds = ['training']
        self.val_folds = ['validation']
        self.test_folds = ['testing']

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.train_folds),
            batch_size=self.batch_size_train,
            num_workers=self.num_workers_train,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """Return the val dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.val_folds),
            batch_size=self.batch_size_val,
            num_workers=self.num_workers_val,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=self.shuffle_validation
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.test_folds),
            batch_size=self.batch_size_test,
            num_workers=self.num_workers_test,
            pin_memory=self.pin_memory,
            shuffle=False
        )

#%%


