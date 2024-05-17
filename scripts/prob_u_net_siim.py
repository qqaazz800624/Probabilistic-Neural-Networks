#%%

from segmentation_models_pytorch import Unet, DeepLabV3Plus

from functools import partial

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary

#from prob_unet import ProbUNet
from prob_unet_proposed import ProbUNet

from siim_ProbNet_datamodule import SIIMDataModule

my_temp_dir = 'results/'


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 128
model_name = 'DeepLabV3Plus'  # Valid model_name: ['Unet', 'DeepLabV3Plus']
latent_dim = 6
beta = 10
batch_size_train = 16
loss_fn = 'DiceLoss'  # Valid loss_fn: ['BCEWithLogitsLoss', 'DiceLoss']

# =========================================== #


if model_name == 'Unet':
    model = Unet(in_channels=1, 
                classes=1, 
                encoder_name = 'tu-resnest50d', 
                encoder_weights = 'imagenet')
    model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/lightning_logs/version_1/checkpoints/best_model.ckpt'

elif model_name == 'DeepLabV3Plus':
    model = DeepLabV3Plus(in_channels=1, 
                        classes=1, 
                        encoder_name = 'tu-resnest50d', 
                        encoder_weights = 'imagenet')
    model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/lightning_logs/version_11/checkpoints/best_model.ckpt'

model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(model_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    model_weight[k_new] = model_weight.pop(k)

model.load_state_dict(model_weight)

#%%


Prob_UNet = ProbUNet(
    model=model,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn
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
    log_every_n_steps=3,
    enable_checkpointing=True,
    enable_progress_bar=True,
    default_root_dir=my_temp_dir,
    num_sanity_val_steps=2,
    callbacks=[lr_monitor, 
               checkpoint_callback,
               model_summarizer]
)

trainer.fit(Prob_UNet, data_module)