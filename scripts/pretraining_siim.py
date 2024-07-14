#%%

from siim_datamodule import SIIMDataModule
#from siim_datamodule_balancedSampler import SIIMDataModule
from lightning.pytorch import Trainer
from deeplabv3plusmodule import DeepLabV3PlusModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from unet_lightningmodule import UNetModule
from mednextmodule import MedNeXtModule
import wandb
from lightning.pytorch.strategies import DDPStrategy


# Initialize the model module
#model = DeepLabV3PlusModule()
model = MedNeXtModule()
#model = SegFormerModule(num_classes=1, in_channels=3)
#model = UNetModule(loss_fn_name='DiceFocalLoss') # valid loss_fn_name: ['DiceLoss', 'BCEWithLogitsLoss', 'DiceCELoss', 'DiceFocalLoss]
siim_data_module = SIIMDataModule(batch_size_train=8, batch_size_val=4, batch_size_test=4)
#siim_data_module = SIIMDataModuleSegFormer()
max_epochs = 64

my_temp_dir = 'results/'
logger = TensorBoardLogger(my_temp_dir)

wandb_logger = WandbLogger(log_model=True, 
                           project="SIIM_pneumothorax_segmentation",
                           save_dir=my_temp_dir,
                           version='version_90',
                           name='MedNeXt_large_v90')

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(filename='best_model', 
                                      monitor='val_loss', 
                                      mode='min',
                                      save_last=True,
                                      save_top_k=1)
model_summarizer = ModelSummary(max_depth=2)

# Initialize the trainer
trainer = Trainer(
    accelerator='gpu',
    devices='auto',
    # strategy=DDPStrategy(find_unused_parameters=True),
    # precision=16,
    max_epochs=max_epochs,  # number of epochs we want to train
    #logger=logger,  # log training metrics for later evaluation
    logger=wandb_logger,  # log training metrics for later evaluation
    log_every_n_steps=8,
    enable_checkpointing=True,
    enable_progress_bar=True,
    default_root_dir=my_temp_dir,
    num_sanity_val_steps=2,
    callbacks=[lr_monitor, 
               checkpoint_callback,
               model_summarizer]
)

# Train the model
trainer.fit(model, siim_data_module)