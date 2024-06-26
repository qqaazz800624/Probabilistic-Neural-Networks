#%%
from SegFormer_lightningmodule import SegFormerModule
#from uw_dataset_segformer import DatasetConfig, TrainingConfig
from siim_dataset_segformer import DatasetConfig, TrainingConfig

# Seed everything for reproducibility.
from lightning import seed_everything
seed_everything(42, workers=True)

#from uw_datamodule_segformer import UW_SegFormerDataModule
from siim_datamodule_segformer import SIIMDataModuleSegFormer
import torch 

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning import Trainer

import wandb

#%%

my_temp_dir = 'results/'


# Intialize custom model.
model = SegFormerModule(
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
data_module = SIIMDataModuleSegFormer(
    num_classes=DatasetConfig.NUM_CLASSES,
    img_size=DatasetConfig.IMAGE_SIZE,
    ds_mean=DatasetConfig.MEAN,
    ds_std=DatasetConfig.STD,
    batch_size_train=TrainingConfig.BATCH_SIZE_TRAIN,
    num_workers_train=TrainingConfig.NUM_WORKERS_TRAIN,
    pin_memory=torch.cuda.is_available(),
)


#%%

# Creating ModelCheckpoint callback. 
# We'll save the model on basis of minimum validation loss.
model_checkpoint = ModelCheckpoint(
                    monitor="valid/loss",
                    mode="min",
                    filename="ckpt_{epoch:03d}-vloss_{valid/loss:.4f}",
                    auto_insert_metric_name=False,
                    )
 
# Creating a learning rate monitor callback which will be plotted/added in the default logger.
lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")

# Initialize logger.
wandb_logger = WandbLogger(log_model=True, 
                           project="SIIM_pneumothorax_segmentation",
                           save_dir=my_temp_dir,
                           version='version_11',
                           name='SegFormer_b4_RandRotation')
tensorboard_logger = TensorBoardLogger(my_temp_dir)
early_stopper = EarlyStopping(monitor="valid/loss", patience=20, mode="min")
model_summarizer = ModelSummary(max_depth=2)

#%%

# Initializing the Trainer class object.
trainer = Trainer(
    accelerator="auto",  # Auto select the best hardware accelerator available
    devices="auto",  # Auto select available devices for the accelerator (For eg. mutiple GPUs)
    strategy="auto",  # Auto select the distributed training strategy.
    max_epochs=TrainingConfig.NUM_EPOCHS,  # Maximum number of epoch to train for.
    enable_model_summary=False,  # Disable printing of model summary as we are using torchinfo.
    callbacks=[model_checkpoint, 
               lr_rate_monitor, 
               model_summarizer,
                early_stopper],  # Declaring callbacks to use.
    precision="16-mixed",  # Using Mixed Precision training.
    logger=wandb_logger,
    default_root_dir=my_temp_dir,
    num_sanity_val_steps=2
)
 
# Start training
trainer.fit(model, data_module)

wandb.finish()
#%%



