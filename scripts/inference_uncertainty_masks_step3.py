#%%
from segmentation_models_pytorch import Unet, DeepLabV3Plus
from functools import partial
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from prob_unet_first import ProbUNet_First
from prob_unet_second import ProbUNet_Second
from prob_unet_third import ProbUNet_Third
import os

from siim_datamodule import SIIMDataModule
from siim_dataset import SIIMDataset
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import json
from tqdm import tqdm


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 64
model_name = 'DeepLabV3Plus'  # Valid model_name: ['Unet', 'DeepLabV3Plus']
latent_dim = 6
beta = 10
batch_size_train = 16

# ============ Inference setting ============= #
num_samples = 100
loss_fn = 'BCEWithLogitsLoss'  # Valid loss_fn: ['BCEWithLogitsLoss','DiceLoss', 'DiceCELoss']

root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'

unet = Unet(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
deeplabv3plus = DeepLabV3Plus(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')

version_prev = None
# =========================================== #

ProbUnet_First = ProbUNet_First(
    #model=unet,
    model=deeplabv3plus,
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

version_no = 'version_80'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First.load_state_dict(model_weight)
ProbUnet_First.eval()
ProbUnet_First.requires_grad_(False)

unet_v2 = Unet(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
deeplabv3plus_v2 = DeepLabV3Plus(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')

ProbUnet_First_v2 = ProbUNet_First(
    #model=unet_v2,
    model=deeplabv3plus_v2,
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

version_no = 'version_80'
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

version_no = 'version_82'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Second.load_state_dict(model_weight)
ProbUnet_Second.eval()
ProbUnet_Second.requires_grad_(False)

#%%

unet_v3 = Unet(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
deeplabv3plus_v3 = DeepLabV3Plus(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')

ProbUnet_First_v3 = ProbUNet_First(
    #model=unet_v3,
    model = deeplabv3plus_v3,
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

version_no = 'version_80'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v3.load_state_dict(model_weight)
ProbUnet_First_v3.eval()
ProbUnet_First_v3.requires_grad_(False)

unet_v4 = Unet(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
deeplabv3plus_v4 = DeepLabV3Plus(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')


ProbUnet_First_v4 = ProbUNet_First(
    #model=unet_v4,
    model=deeplabv3plus_v4,
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

version_no = 'version_80'
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

version_no = 'version_82'
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

version_no = 'version_84'
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Third.load_state_dict(model_weight)
ProbUnet_Third.eval()

#%%

#from siim_dataset import SIIMDataset
from siim_dataset_masks import SIIMDataset
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from monai.transforms import MapLabelValue
import matplotlib.pyplot as plt

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/masks'

train_dataset = SIIMDataset(folds=['training'], if_test=True)
mapper = MapLabelValue(orig_labels=[0, 1], target_labels=[0, 255])

ProbUnet_Second.to(device)

for img_serial in tqdm(range(len(train_dataset))):
    image = train_dataset[img_serial]['input']  # shape: [1, 512, 512]
    label = train_dataset[img_serial]['target']  # shape: [1, 512, 512]
    image_basename = train_dataset[img_serial]['basename']
    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_outputs, prior_mu, prior_sigma = ProbUnet_Second.predict_step(input_image)
        stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
        stacked_samples = torch.sigmoid(stacked_samples)
        uncertainty_heatmap = stacked_samples.var(dim=0, keepdim=False)
        mask_uncertainty = torch.zeros_like(uncertainty_heatmap)
        quantile = torch.quantile(uncertainty_heatmap.flatten(), 0.975).item()
        mask_uncertainty = torch.where(uncertainty_heatmap > quantile, torch.ones_like(uncertainty_heatmap), torch.zeros_like(uncertainty_heatmap))
        mask_uncertainty = mask_uncertainty.detach().cpu().numpy().T
        mask_uncertainty = mapper(mask_uncertainty).astype(np.uint8)
        Image.fromarray(mask_uncertainty).save(os.path.join(root_dir, f'{image_basename}'))

#%%

#from siim_dataset import SIIMDataset
from siim_dataset_masks import SIIMDataset
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from monai.transforms import MapLabelValue

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/masks'

ProbUnet_Second.to(device)
val_dataset = SIIMDataset(folds=['validation'], if_test=True)
mapper = MapLabelValue(orig_labels=[0, 1], target_labels=[0, 255])

for img_serial in tqdm(range(len(val_dataset))):
    image = val_dataset[img_serial]['input']  # shape: [1, 512, 512]
    label = val_dataset[img_serial]['target']  # shape: [1, 512, 512]
    image_basename = val_dataset[img_serial]['basename']
    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_outputs, prior_mu, prior_sigma = ProbUnet_Second.predict_step(input_image)
        stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
        stacked_samples = torch.sigmoid(stacked_samples)
        uncertainty_heatmap = stacked_samples.var(dim=0, keepdim=False)
        mask_uncertainty = torch.zeros_like(uncertainty_heatmap)
        quantile = torch.quantile(uncertainty_heatmap.flatten(), 0.975).item()
        mask_uncertainty = torch.where(uncertainty_heatmap > quantile, torch.ones_like(uncertainty_heatmap), torch.zeros_like(uncertainty_heatmap))
        mask_uncertainty = mask_uncertainty.detach().cpu().numpy().T
        mask_uncertainty = mapper(mask_uncertainty).astype(np.uint8)
        Image.fromarray(mask_uncertainty).save(os.path.join(root_dir, f'{image_basename}'))


#%%
