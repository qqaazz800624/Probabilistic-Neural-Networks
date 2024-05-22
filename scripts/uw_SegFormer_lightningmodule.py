#%%

from uw_dataset_segformer import DatasetConfig, InferenceConfig
from uw_datamodule_segformer import MedicalSegmentationDataModule

#%%

dm = MedicalSegmentationDataModule(
    num_classes=DatasetConfig.NUM_CLASSES,
    img_size=DatasetConfig.IMAGE_SIZE,
    ds_mean=DatasetConfig.MEAN,
    ds_std=DatasetConfig.STD,
    batch_size=InferenceConfig.BATCH_SIZE,
    num_workers=2,
    shuffle_validation=True,
)

# Donwload dataset.
dm.prepare_data()
 
# Create training & validation dataset.
dm.setup()
 
train_loader, valid_loader = dm.train_dataloader(), dm.val_dataloader()


#%%






#%%







#%%









#%%










#%%