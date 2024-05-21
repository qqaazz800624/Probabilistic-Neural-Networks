#%%

from segmentation_models_pytorch import Unet, DeepLabV3Plus

from functools import partial

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from hierarchical_prob_unet import HierarchicalProbUNet
from siim_ProbNet_datamodule import SIIMDataModule

my_temp_dir = 'results/'


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 32
batch_size_train = 16
loss_type = 'geco'  # valid loss_types = ["elbo", "geco"]

# =========================================== #

HP_UNet = HierarchicalProbUNet(
    num_in_channels=1,
    num_classes=1,
    task='binary',
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    loss_type=loss_type
)

data_module = SIIMDataModule(batch_size_train=batch_size_train)

logger = TensorBoardLogger(my_temp_dir)
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(filename='best_model', 
                                      monitor='val_loss', 
                                      mode='min',
                                      save_last=True,
                                      save_top_k=1)
model_summarizer = ModelSummary(max_depth=2)


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

trainer.fit(HP_UNet, data_module)