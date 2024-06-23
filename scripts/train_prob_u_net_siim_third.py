#%%

from segmentation_models_pytorch import Unet

from functools import partial

import os
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.strategies import DDPStrategy

from prob_unet_first import ProbUNet_First
from prob_unet_second import ProbUNet_Second
from prob_unet_third import ProbUNet_Third

from siim_ProbNet_datamodule import SIIMDataModule
import wandb

my_temp_dir = 'results/'


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 64
model_name = 'Unet'  # Valid model_name: ['Unet', 'DeepLabV3Plus']
latent_dim = 6
beta = 10
batch_size_train = 16
batch_size_val = 16
loss_fn = 'BCEWithLogitsLoss'  # Valid loss_fn: ['BCEWithLogitsLoss', 'DiceLoss', 'DiceCELoss']

# =========================================== #

root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'

#%% Load the trained model: ProbUNet_Second
unet = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
# model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
# unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
# for k in list(unet_weight.keys()):
#     k_new = k.replace(
#         "model.", "", 1
#     )  # e.g. "model.conv.weight" => conv.weight"
#     unet_weight[k_new] = unet_weight.pop(k)
# unet.load_state_dict(unet_weight)

version_prev = None

ProbUnet_First = ProbUNet_First(
    model=unet,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_35'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First.load_state_dict(model_weight)
ProbUnet_First.eval()
ProbUnet_First.requires_grad_(False)

unet_v2 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
# model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
# unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
# for k in list(unet_weight.keys()):
#     k_new = k.replace(
#         "model.", "", 1
#     )  # e.g. "model.conv.weight" => conv.weight"
#     unet_weight[k_new] = unet_weight.pop(k)
# unet_v2.load_state_dict(unet_weight)

ProbUnet_First_v2 = ProbUNet_First(
    model=unet_v2,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_35'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v2.load_state_dict(model_weight)

model_first = ProbUnet_First_v2.model
prior_first = ProbUnet_First_v2.prior
posterior_first = ProbUnet_First_v2.posterior
fcomb_first = ProbUnet_First_v2.fcomb

ProbUnet_Second = ProbUNet_Second(
    model=model_first,
    prior_first=prior_first,
    posterior_first=posterior_first,
    fcomb_first=fcomb_first,
    prob_unet_first=ProbUnet_First,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_51'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Second.load_state_dict(model_weight)
ProbUnet_Second.eval()
ProbUnet_Second.requires_grad_(False)

#%% Obtain the modules: model_second, prior_second, posterior_second, fcomb_second

unet_v3 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
# model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
# unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
# for k in list(unet_weight.keys()):
#     k_new = k.replace(
#         "model.", "", 1
#     )  # e.g. "model.conv.weight" => conv.weight"
#     unet_weight[k_new] = unet_weight.pop(k)
# unet_v3.load_state_dict(unet_weight)

ProbUnet_First_v3 = ProbUNet_First(
    model=unet_v3,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_35'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v3.load_state_dict(model_weight)
ProbUnet_First_v3.eval()
ProbUnet_First_v3.requires_grad_(False)

unet_v4 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
# model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
# unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
# for k in list(unet_weight.keys()):
#     k_new = k.replace(
#         "model.", "", 1
#     )  # e.g. "model.conv.weight" => conv.weight"
#     unet_weight[k_new] = unet_weight.pop(k)
# unet_v4.load_state_dict(unet_weight)

ProbUnet_First_v4 = ProbUNet_First(
    model=unet_v4,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_35'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v4.load_state_dict(model_weight)

model_first_v4 = ProbUnet_First_v4.model
prior_first_v4 = ProbUnet_First_v4.prior
posterior_first_v4 = ProbUnet_First_v4.posterior
fcomb_first_v4 = ProbUnet_First_v4.fcomb

ProbUnet_Second_v2 = ProbUNet_Second(
    model=model_first_v4,
    prior_first=prior_first_v4,
    posterior_first=posterior_first_v4,
    fcomb_first=fcomb_first_v4,
    prob_unet_first=ProbUnet_First_v3,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_51'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Second_v2.load_state_dict(model_weight)

model_second = ProbUnet_Second_v2.model
prior_second = ProbUnet_Second_v2.prior
posterior_second = ProbUnet_Second_v2.posterior
fcomb_second = ProbUnet_Second_v2.fcomb

#%%

ProbUnet_Third = ProbUNet_Third(
    model=model_second,
    prior_second=prior_second,
    posterior_second=posterior_second,
    fcomb_second=fcomb_second,
    prob_unet_second=ProbUnet_Second,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

#%%

data_module = SIIMDataModule(batch_size_train=batch_size_train,
                             batch_size_val=batch_size_val)

logger = TensorBoardLogger(my_temp_dir)
wandb_logger = WandbLogger(log_model=True, 
                           project="SIIM_pneumothorax_segmentation",
                           save_dir=my_temp_dir,
                           version='version_53',
                           name='ProbUNet_step3_64epochs_v53')

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
    #strategy=DDPStrategy(find_unused_parameters=True),
    #precision=16,
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

trainer.fit(ProbUnet_Third, data_module)

wandb.finish()
#%%