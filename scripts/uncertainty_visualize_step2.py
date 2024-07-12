#%%

from segmentation_models_pytorch import Unet, DeepLabV3Plus
from functools import partial
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from prob_unet_first import ProbUNet_First
#from prob_unet_second import ProbUNet_Second
from prob_unet_second_masks import ProbUNet_Second
import os

from siim_datamodule import SIIMDataModule
#from siim_ProbUNet_datamodule_masks import SIIMDataModule
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
batch_size_val = 16

# ============ Inference setting ============= #

num_samples = 100
loss_fn = 'BCEWithLogitsLoss'  # Valid loss_fn: ['BCEWithLogitsLoss','DiceLoss', 'DiceCELoss']

root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'

unet = Unet(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
deeplabv3plus = DeepLabV3Plus(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')

# =========================================== #

version_prev = None

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

#%%

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


#%%

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
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Second.load_state_dict(model_weight)
ProbUnet_Second.eval()

#%%

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

data_module = SIIMDataModule(batch_size_test=1, num_workers_test=2)
test_data_loader = data_module.test_dataloader()
dice_metric = DiceMetric(include_background=True, reduction='none', ignore_empty=False)
discreter = AsDiscrete(threshold=0.5)

ProbUnet_Second.to(device)

dice_scores = []

with torch.no_grad():
    for data in tqdm(test_data_loader):
        img, label = data['input'].to(device), data['target']
        prediction_outputs, prior_mu, prior_sigma = ProbUnet_Second.predict_step(img)
        stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
        stacked_samples = torch.sigmoid(stacked_samples)
        prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)
        dice_metric(y_pred=discreter(prediction_heatmap.cpu().unsqueeze(0).unsqueeze(0)), y=discreter(label.unsqueeze(0)))
        dice_score = dice_metric.aggregate().item()
        dice_scores.append(dice_score)
        dice_metric.reset()

print('Dice score: ', sum(dice_scores)/len(dice_scores))

with open(f'results/dice_scores_ProbUnet_step2.json', 'w') as file:
    json.dump(dice_scores, file)

#%% Single image dice evaluation

# import os
# from siim_dataset import SIIMDataset

# fold_no = 'testing'
# # Good: 6, 92, 522, 212, 207 Bad: 532, 484, 168
# # large mask: 92, 417, 492, 339, 132, 302
# # large-medium mask: 338, 377
# # medium mask: 107, 136
# # medium-small mask: 29, 412
# # small mask: 128, 184
# img_serial = 184
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# test_dataset = SIIMDataset(folds=[fold_no], if_test=True)

# image = test_dataset[img_serial]['input']  # shape: [1, 512, 512]
# mask = test_dataset[img_serial]['target']  # shape: [1, 512, 512]

# input_image = image.unsqueeze(0).to(device)
# ProbUnet_Second.to(device)
# with torch.no_grad():
#     prediction_outputs, prior_mu, prior_sigma = ProbUnet_Second.predict_step(input_image)

# stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
# stacked_samples = torch.sigmoid(stacked_samples)
# prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)
# prediction_heatmap = prediction_heatmap.cpu()

# discreter = AsDiscrete(threshold=0.5)
# dice_metric = DiceMetric(include_background=True, reduction='mean')
# dice_metric(y_pred=discreter(prediction_heatmap.unsqueeze(0).unsqueeze(0)), y=discreter(mask.unsqueeze(0)))
# dice_score = dice_metric.aggregate().item()
# dice_metric.reset()

# print('Dice score: ', dice_score)

# #%% Prediction heatmap

# import matplotlib.pyplot as plt
# plt.imshow(prediction_heatmap.detach().numpy().T, 
#            cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Prediction heatmap: {fold_no}_{img_serial}')

# #%% Uncertainty heatmap

# stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
# stacked_samples = torch.sigmoid(stacked_samples)
# uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim=False)

# import matplotlib.pyplot as plt

# plt.imshow(uncertainty_heatmap.cpu().detach().numpy().T, 
#            cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Heatmap of Epistemic Uncertainty: {fold_no}_{img_serial}')

# #%% uncertainty mask

# input_image = image.unsqueeze(0).to(device)
# ProbUnet_First.to(device)
# with torch.no_grad():
#     #prediction_outputs, prior_mu, prior_sigma = ProbUnet_First.predict_step(input_image)
#     prediction_outputs, prior_mu, prior_sigma = ProbUnet_Second.predict_step(input_image)
#     stacked_samples = torch.sigmoid(prediction_outputs['samples'])
#     uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim = False)
#     uncertainty_heatmap = uncertainty_heatmap.squeeze(0).squeeze(0)


# # Thresholding the uncertainty heatmap
# quantile = torch.quantile(uncertainty_heatmap, 0.975).item()
# #print("97.5th quantile of uncertainty_heatmap:", quantile)
# mask_uncertainty = torch.where(uncertainty_heatmap.cpu() > quantile, 
#                                torch.ones_like(mask.squeeze(0)), 
#                                torch.zeros_like(mask.squeeze(0)))

# import torch
# import matplotlib.pyplot as plt

# plt.imshow(mask_uncertainty.detach().numpy().T, 
#            cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Uncertainty mask: {fold_no}_{img_serial}')

#%%
