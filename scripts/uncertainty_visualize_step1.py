#%%

from segmentation_models_pytorch import Unet, DeepLabV3Plus
from custom.mednext import mednext_base, mednext_large

from functools import partial
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from prob_unet_first import ProbUNet_First
import os

from siim_datamodule import SIIMDataModule
from siim_dataset import SIIMDataset
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import json
from tqdm import tqdm


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 128
model_name = 'mednext'  # Valid model_name: ['Unet', 'DeepLabV3Plus', 'mednext']
latent_dim = 6
beta = 10
batch_size_train = 16

# ============ Inference setting ============= #
num_samples = 100
loss_fn = 'BCEWithLogitsLoss'  # Valid loss_fn: ['BCEWithLogitsLoss','DiceLoss', 'DiceCELoss']

unet = Unet(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
deeplabv3plus = DeepLabV3Plus(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
#mednext = mednext_base(in_channels=1, out_channels=1, spatial_dims=2, use_grad_checkpoint=True)
mednext = mednext_large(in_channels=1, out_channels=1, spatial_dims=2, use_grad_checkpoint=True)

# =========================================== #

version_prev = None

ProbUnet_First = ProbUNet_First(
    #model=unet,
    #model=deeplabv3plus,
    model = mednext,
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

version_no = 'version_91'
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First.load_state_dict(model_weight)
ProbUnet_First.eval()

#%%
# testing dataset
# number of labeled data: 535
# number of unlabeled data: 1876
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

data_module = SIIMDataModule(batch_size_test=1, num_workers_test=2)
test_data_loader = data_module.test_dataloader()
dice_metric = DiceMetric(include_background=True, reduction='none', ignore_empty=False)
discreter = AsDiscrete(threshold=0.5)

ProbUnet_First.to(device)

dice_scores = []

with torch.no_grad():
    for data in tqdm(test_data_loader):
        img, label = data['input'].to(device), data['target']
        prediction_outputs, prior_mu, prior_sigma = ProbUnet_First.predict_step(img)
        stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
        stacked_samples = torch.sigmoid(stacked_samples)
        prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)
        dice_metric(y_pred=discreter(prediction_heatmap.cpu().unsqueeze(0).unsqueeze(0)), y=discreter(label.unsqueeze(0)))
        dice_score = dice_metric.aggregate().item()
        dice_scores.append(dice_score)
        dice_metric.reset()

print('Dice score: ', sum(dice_scores)/len(dice_scores))

with open(f'results/dice_scores_ProbUnet_step1.json', 'w') as file:
    json.dump(dice_scores, file)

#%%

import json

with open('../results/dice_scores_ProbUnet_step3_train.json', 'r') as file:
    dice_scores_ProbUnet_step3_train = json.load(file)

with open('../results/dice_scores_ProbUnet_step3_val.json', 'r') as file:
    dice_scores_ProbUnet_step3_val = json.load(file)

with open('../results/dice_scores_ProbUnet_step3_test.json', 'r') as file:
    dice_scores_ProbUnet_step3_test = json.load(file)

# with open('../results/dice_scores_ProbUnet_step3_v62.json', 'r') as file:
#     dice_scores_ProbUnet_step3_v62 = json.load(file)

# with open('../results/dice_scores_ProbUnet_step3_v63.json', 'r') as file:
#     dice_scores_ProbUnet_step3_v63 = json.load(file)

# with open('../results/dice_scores_ProbUnet_step4_v64.json', 'r') as file:
#     dice_scores_ProbUnet_step4_v64 = json.load(file)

# with open('../results/dice_scores_ProbUnet_step4_v65.json', 'r') as file:
#     dice_scores_ProbUnet_step4_v65 = json.load(file)

# with open('../results/dice_scores_Unet.json', 'r') as file:
#     dice_scores_Unet = json.load(file)



#%%

import numpy as np

dice_scores_ProbUnet_step3_train = np.array(dice_scores_ProbUnet_step3_train)
dice_scores_ProbUnet_step3_val = np.array(dice_scores_ProbUnet_step3_val)
dice_scores_ProbUnet_step3_test = np.array(dice_scores_ProbUnet_step3_test)

print(dice_scores_ProbUnet_step3_train.shape)
print(dice_scores_ProbUnet_step3_val.shape)
print(dice_scores_ProbUnet_step3_test.shape)

#%%

# dice_scores_ProbUnet_step1 = np.array(dice_scores_ProbUnet_step1)
# dice_scores_ProbUnet_step2_v60 = np.array(dice_scores_ProbUnet_step2_v60)
# dice_scores_ProbUnet_step2_v61 = np.array(dice_scores_ProbUnet_step2_v61)
# dice_scores_ProbUnet_step3_v62 = np.array(dice_scores_ProbUnet_step3_v62)
# dice_scores_ProbUnet_step3_v63 = np.array(dice_scores_ProbUnet_step3_v63)
# dice_scores_ProbUnet_step4_v64 = np.array(dice_scores_ProbUnet_step4_v64)
# dice_scores_ProbUnet_step4_v65 = np.array(dice_scores_ProbUnet_step4_v65)

# dice_scores_ProbUnet_step1 = np.round(np.array(dice_scores_ProbUnet_step1),3)
# dice_scores_ProbUnet_step2 = np.round(np.array(dice_scores_ProbUnet_step2),3)
# dice_scores_ProbUnet_step3 = np.round(np.array(dice_scores_ProbUnet_step3),3)
# dice_scores_ProbUnet_step4 = np.round(np.array(dice_scores_ProbUnet_step4),3)
# dice_scores_Unet = np.round(np.array(dice_scores_Unet),3)
# labeled_scores_step1 = dice_scores_ProbUnet_step1[0:535]
# labeled_scores_step2 = dice_scores_ProbUnet_step2[0:535]
# labeled_scores_step3 = dice_scores_ProbUnet_step3[0:535]
# labeled_scores_step4 = dice_scores_ProbUnet_step4[0:535]
# labeled_scores_Unet = dice_scores_Unet[0:535]
# unlabeled_scores_step1 = dice_scores_ProbUnet_step1[535:]
# unlabeled_scores_step2 = dice_scores_ProbUnet_step2[535:]
# unlabeled_scores_step3 = dice_scores_ProbUnet_step3[535:]
# unlabeled_scores_step4 = dice_scores_ProbUnet_step4[535:]
# unlabeled_scores_Unet = dice_scores_Unet[535:]

#%%

dice_scores_ProbUnet_step3_train[0:50]


#%%

dice_scores_ProbUnet_step3_train.mean()
# unlabeled_scores_Unet.mean()

#%%
import matplotlib.pyplot as plt

# Draw histogram for dice_scores_Unet
plt.hist(dice_scores_ProbUnet_step3_train, bins=10, edgecolor='black', alpha=0.7, label='dice_scores_ProbUnet_step3_train',color='lightblue')

# Draw histogram for labeled_scores_step1
plt.hist(dice_scores_ProbUnet_step3_val, bins=10, edgecolor='black', alpha=0.5, label='dice_scores_ProbUnet_step3_val',color='orange')

# Draw histogram for labeled_scores_step1_192epochs_v46
plt.hist(dice_scores_ProbUnet_step3_test, bins=10, edgecolor='black', alpha=0.3, label='dice_scores_ProbUnet_step3_test', color='yellow')


# Draw histogram for labeled_scores_step2
#plt.hist(labeled_scores_step2, bins=10, edgecolor='black', alpha=0.4, label='labeled_scores_step2', color='green')

# Draw histogram for labeled_scores_step3
#plt.hist(labeled_scores_step3, bins=10, edgecolor='black', alpha=0.5, label='labeled_scores_step3', color='yellow')

# Draw histogram for labeled_scores_step4
#plt.hist(labeled_scores_step4, bins=10, edgecolor='black', alpha=0.6, label='labeled_scores_step4', color='gray')


# Add labels and title
plt.xlabel('Dice Score')
plt.ylabel('Frequency')
plt.title('Histogram of Dice Scores')

# Add legend
plt.legend()

# Show the histogram
plt.show()

#%% Single image dice evaluation

import os
from siim_dataset import SIIMDataset

fold_no = 'testing'
# Good: 6, 92, 522, 292, 207, 212 Bad: 532, 484, 168
# large mask: 92, 417, 492, 339, 132, 302
# large-medium mask: 338, 377, 325
# medium mask: 107, 136
# medium-small mask: 29, 412
# small mask: 128, 184
img_serial = 168
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

test_dataset = SIIMDataset(folds=[fold_no], if_test=True)

image = test_dataset[img_serial]['input']  # shape: [1, 512, 512]
mask = test_dataset[img_serial]['target']  # shape: [1, 512, 512]

input_image = image.unsqueeze(0).to(device)
ProbUnet_First.to(device)
with torch.no_grad():
    prediction_outputs, prior_mu, prior_sigma = ProbUnet_First.predict_step(input_image)

stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
stacked_samples = torch.sigmoid(stacked_samples)
prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)
prediction_heatmap = prediction_heatmap.cpu()

discreter = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction='mean')
dice_metric(y_pred=discreter(prediction_heatmap.unsqueeze(0).unsqueeze(0)), y=discreter(mask.unsqueeze(0)))
dice_score = dice_metric.aggregate().item()
dice_metric.reset()

print('Dice score: ', dice_score)

#%% Prediction heatmap

import matplotlib.pyplot as plt
plt.imshow(prediction_heatmap.detach().numpy().T, 
           cmap='plasma', aspect='auto', vmin=0, vmax=1)
plt.colorbar()
plt.title(f'Prediction heatmap: {fold_no}_{img_serial}')

#%% Uncertainty heatmap

stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
stacked_samples = torch.sigmoid(stacked_samples)
uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim=False)

import matplotlib.pyplot as plt

plt.imshow(uncertainty_heatmap.cpu().detach().numpy().T, 
           cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Heatmap of Epistemic Uncertainty: {fold_no}_{img_serial}')


#%%
