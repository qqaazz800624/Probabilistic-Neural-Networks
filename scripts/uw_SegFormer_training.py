#%%
from uw_SegFormer_lightningmodule import UW_SegFormerModule
from uw_dataset_segformer import DatasetConfig, TrainingConfig
# Seed everything for reproducibility.
from lightning import seed_everything
seed_everything(42, workers=True)
from uw_datamodule_segformer import UW_SegFormerDataModule
import torch 

#%%

# Intialize custom model.
model = UW_SegFormerModule(
    model_name=TrainingConfig.MODEL_NAME,
    num_classes=DatasetConfig.NUM_CLASSES,
    init_lr=TrainingConfig.INIT_LR,
    optimizer_name=TrainingConfig.OPTIMIZER_NAME,
    weight_decay=TrainingConfig.WEIGHT_DECAY,
    use_scheduler=TrainingConfig.USE_SCHEDULER,
    scheduler_name=TrainingConfig.SCHEDULER,
    num_epochs=TrainingConfig.NUM_EPOCHS,
) 


# Initialize custom data module.
data_module = UW_SegFormerDataModule(
    num_classes=DatasetConfig.NUM_CLASSES,
    img_size=DatasetConfig.IMAGE_SIZE,
    ds_mean=DatasetConfig.MEAN,
    ds_std=DatasetConfig.STD,
    batch_size=TrainingConfig.BATCH_SIZE,
    num_workers=TrainingConfig.NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


#%%


from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Creating ModelCheckpoint callback. 
# We'll save the model on basis on validation f1-score.
model_checkpoint = ModelCheckpoint(
    monitor="valid/f1",
    mode="max",
    filename="ckpt_{epoch:03d}-vloss_{valid/loss:.4f}_vf1_{valid/f1:.4f}",
    auto_insert_metric_name=False,
)
 
# Creating a learning rate monitor callback which will be plotted/added in the default logger.
lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")

# Initialize logger.
wandb_logger = WandbLogger(log_model=True, project="UM_medical_segmentation")



#%%


from lightning import Trainer

# Initializing the Trainer class object.
trainer = Trainer(
    accelerator="auto",  # Auto select the best hardware accelerator available
    devices="auto",  # Auto select available devices for the accelerator (For eg. mutiple GPUs)
    strategy="auto",  # Auto select the distributed training strategy.
    max_epochs=TrainingConfig.NUM_EPOCHS,  # Maximum number of epoch to train for.
    enable_model_summary=False,  # Disable printing of model summary as we are using torchinfo.
    callbacks=[model_checkpoint, lr_rate_monitor],  # Declaring callbacks to use.
    precision="16-mixed",  # Using Mixed Precision training.
    logger=wandb_logger
)
 
# Start training
trainer.fit(model, data_module)





#%%








#%%









#%%