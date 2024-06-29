#%%

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from siim_dataset_masks import SIIMDataset

from monai.transforms import Compose, RandFlipd, RandAdjustContrastd, RandShiftIntensityd, Rand2DElasticd, RandAffined, RandGridDistortiond
from monai.transforms import EnsureTyped, AsDiscreted, NormalizeIntensityd, Resized
from custom.augmentations_masks import XRayAugs
from custom.balanced_data_loader import balanced_data_loader
from custom.monai_OneOf import OneOf, OneOfd

class SIIMDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size_train = 8,
        num_workers_train = 2,
        batch_size_val = 4,
        num_workers_val = 2,
        batch_size_test = 4,
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
                                XRayAugs(img_key='image', seg_key='target', mask_key='mask'),
                                # RandFlipd(keys=['image', 'target', 'mask'], prob=0.5, spatial_axis=1),
                                # OneOfd(keys=['image'], 
                                #        transforms=[
                                #         RandAdjustContrastd(keys=['image'], gamma=(0.5, 1.5), prob=1.0),
                                #         RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
                                #         ], 
                                #         weights=[0.5, 0.5],
                                #         prob=0.3),
                                # OneOfd(keys=['image', 'target', 'mask'],
                                #         transforms=[
                                #             Rand2DElasticd(keys=['image', 'target', 'mask'], prob=1.0, magnitude_range=(1, 2), shear_range=(0.03, 0.03), rotate_range=(0.0, 0.0),
                                #                            spacing=(20, 20), padding_mode='zeros', mode=['bilinear', 'nearest', 'nearest']),
                                #             RandGridDistortiond(keys=['image', 'target', 'mask'], prob=1.0, distort_limit=(-0.2, 0.2), 
                                #                                 num_cells=4, mode=['bilinear', 'nearest', 'nearest']),
                                #             RandAffined(keys=['image', 'target', 'mask'], prob=1.0, shear_range=(0.5, 0.5), translate_range=(0.5, 0.5), scale_range=(0.9, 1.1),
                                #                         mode=['bilinear', 'nearest', 'nearest'])
                                #             ],
                                #         weights=[0.33, 0.33, 0.34],
                                #         prob=0.3),
                                # RandAffined(keys=['image', 'target', 'mask'], prob=1, shear_range=(0.5, 0.5), mode=['bilinear', 'nearest', 'nearest'], padding_mode='zeros'),
                                # Resized(keys=['image', 'target', 'mask'], spatial_size=(512, 512)),
                                NormalizeIntensityd(keys=['image']),
                                AsDiscreted(keys=['target', 'mask'], threshold=0.5),
                                EnsureTyped(keys=['image', 'target', 'mask'], dtype='float32')
                                ])
        
        self.val_transforms = Compose([
                                NormalizeIntensityd(keys=['image']),
                                AsDiscreted(keys=['target', 'mask'], threshold=0.5),
                                EnsureTyped(keys=['image', 'target', 'mask'], dtype='float32')
                                ])
        
        self.test_transforms = Compose([
                                NormalizeIntensityd(keys=['image']),
                                AsDiscreted(keys=['target', 'mask'], threshold=0.5),
                                EnsureTyped(keys=['image', 'target', 'mask'], dtype='float32')
                                ])

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.train_folds, transform=self.train_transforms),
            batch_size=self.batch_size_train,
            num_workers=self.num_workers_train,
            shuffle=False,
            drop_last=True
        )
    
    # def train_dataloader(self) -> DataLoader:
    #     """Return the train dataloader."""
    #     return balanced_data_loader(
    #         SIIMDataset(folds=self.train_folds, transform=self.train_transforms),
    #         batch_size=self.batch_size_train,
    #         num_workers=self.num_workers_train,
    #     )

    def val_dataloader(self):
        """Return the val dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.val_folds, transform=self.val_transforms),
            batch_size=self.batch_size_val,
            num_workers=self.num_workers_val,
            shuffle=False,
            drop_last=True
        )
    
    # def val_dataloader(self) -> DataLoader:
    #     """Return the val dataloader."""
    #     return balanced_data_loader(
    #         SIIMDataset(folds=self.val_folds, transform=self.val_transforms),
    #         batch_size=self.batch_size_val,
    #         num_workers=self.num_workers_val,
    #     )

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            SIIMDataset(folds=self.test_folds, transform=self.test_transforms),
            batch_size=self.batch_size_test,
            num_workers=self.num_workers_test,
            shuffle=False,
            drop_last=True
        )
    
    # def test_dataloader(self) -> DataLoader:
    #     """Return the test dataloader."""
    #     return balanced_data_loader(
    #         SIIMDataset(folds=self.test_folds, transform=self.test_transforms),
    #         batch_size=self.batch_size_test,
    #         num_workers=self.num_workers_test,
    #     )

#%%





#%%