#%%

from siim_datamodule import SIIMDataModule
from siim_datamodule_segformer import SIIMDataModuleSegFormer
from lightning.pytorch import Trainer
from deeplabv3plusmodule import DeepLabV3PlusModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from unet_lightningmodule import UNetModule

# Initialize the model module
model = DeepLabV3PlusModule(in_channels=1)
#model = UNetModule()
siim_data_module = SIIMDataModule()
#siim_data_module = SIIMDataModuleSegFormer()
max_epochs = 64

my_temp_dir = 'results/'
logger = TensorBoardLogger(my_temp_dir)
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
    devices=1,
    max_epochs=max_epochs,  # number of epochs we want to train
    logger=logger,  # log training metrics for later evaluation
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