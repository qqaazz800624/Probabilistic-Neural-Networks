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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
# ============ Training setting ============= #

max_epochs = 32
batch_size_train = 16
loss_type = 'elbo'  # valid loss_types = ["elbo", "geco"]

# =========================================== #

model = Unet(in_channels=1, 
                classes=1, 
                encoder_name = 'tu-resnest50d', 
                encoder_weights = 'imagenet')

model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/lightning_logs/version_1/checkpoints/best_model.ckpt'

model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(model_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    model_weight[k_new] = model_weight.pop(k)

model.load_state_dict(model_weight)


#%%

HP_UNet = HierarchicalProbUNet(
    #model=model,
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

#%%

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

trainer.fit(HP_UNet.to(device), data_module)
# %%
