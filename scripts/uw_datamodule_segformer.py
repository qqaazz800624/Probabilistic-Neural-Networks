#%%

from lightning import LightningDataModule
from uw_dataset_segformer import DatasetConfig, Paths, UW_SegformerDataset
import os
import requests
import zipfile
from glob import glob
from torch.utils.data import DataLoader

#%%

class UW_SegFormerDataModule(LightningDataModule):
    def __init__(
        self,
        num_classes=10,
        img_size=(384, 384),
        ds_mean=(0.485, 0.456, 0.406),
        ds_std=(0.229, 0.224, 0.225),
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        shuffle_validation=False,
    ):
        super().__init__()
 
        self.num_classes = num_classes
        self.img_size    = img_size
        self.ds_mean     = ds_mean
        self.ds_std      = ds_std
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
         
        self.shuffle_validation = shuffle_validation
 
    def prepare_data(self):
        # Download dataset.
        dataset_zip_path = f"{DatasetConfig.DATASET_PATH}.zip"
 
        # Download if dataset does not exists.
        if not os.path.exists(DatasetConfig.DATASET_PATH):
 
            print("Downloading and extracting assets...", end="")
            file = requests.get(DatasetConfig.URL)
            open(dataset_zip_path, "wb").write(file.content)
 
            try:
                with zipfile.ZipFile(dataset_zip_path) as z:
                    z.extractall(os.path.split(dataset_zip_path)[0]) # Unzip where downloaded.
                    print("Done")
            except:
                print("Invalid file")
 
            os.remove(dataset_zip_path) # Remove the ZIP file to free storage space.
 
    def setup(self, *args, **kwargs):
        # Create training dataset and dataloader.
        train_imgs = sorted(glob(f"{Paths.DATA_TRAIN_IMAGES}"))
        train_msks  = sorted(glob(f"{Paths.DATA_TRAIN_LABELS}"))
 
        # Create validation dataset and dataloader.
        valid_imgs = sorted(glob(f"{Paths.DATA_VALID_IMAGES}"))
        valid_msks = sorted(glob(f"{Paths.DATA_VALID_LABELS}"))
 
        self.train_ds = UW_SegformerDataset(image_paths=train_imgs, mask_paths=train_msks, img_size=self.img_size,  
                                       is_train=True, ds_mean=self.ds_mean, ds_std=self.ds_std)
 
        self.valid_ds = UW_SegformerDataset(image_paths=valid_imgs, mask_paths=valid_msks, img_size=self.img_size, 
                                       is_train=False, ds_mean=self.ds_mean, ds_std=self.ds_std)
 
    def train_dataloader(self):
        # Create train dataloader object with drop_last flag set to True.
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory, 
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )    
 
    def val_dataloader(self):
        # Create validation dataloader object.
        return DataLoader(
            self.valid_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory, 
            num_workers=self.num_workers, shuffle=self.shuffle_validation
        )