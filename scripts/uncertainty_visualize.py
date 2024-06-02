#%%

from segmentation_models_pytorch import DeepLabV3Plus, Unet
from functools import partial
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from prob_unet import ProbUNet
import os
#from utils import image_preprocessor
from siim_datamodule import SIIMDataModule
from siim_dataset import SIIMDataset
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import json
from tqdm import tqdm


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 128
model_name = 'Unet'  # Valid model_name: ['Unet', 'DeepLabV3Plus']
latent_dim = 6
beta = 10
batch_size_train = 12
# model_version_dict = {'Unet': 'version_6',
#                       'DeepLabV3Plus': 'version_7'}

# ============ Inference setting ============= #
num_samples = 30
loss_fn = 'DiceLoss'  # Valid loss_fn: ['BCEWithLogitsLoss','DiceLoss']

if model_name == 'Unet':
    model = Unet(in_channels=1, 
                classes=1, 
                encoder_name = 'tu-resnest50d', 
                encoder_weights = 'imagenet')

elif model_name == 'DeepLabV3Plus':
    model = DeepLabV3Plus(in_channels=1, 
                        classes=1, 
                        encoder_name = 'tu-resnest50d', 
                        encoder_weights = 'imagenet')

# =========================================== #

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
    loss_fn=loss_fn,
    num_samples=num_samples
    )

#version_no = model_version_dict[model_name]
version_no = 'version_15'
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
Prob_UNet.load_state_dict(model_weight)
Prob_UNet.eval()

#%%

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

data_module = SIIMDataModule(batch_size_test=1, num_workers_test=2)
test_data_loader = data_module.test_dataloader()
dice_metric = DiceMetric(include_background=True, reduction='none', ignore_empty=False)
discreter = AsDiscrete(threshold=0.5)

Prob_UNet.to(device)

dice_scores = []

with torch.no_grad():
    for data in tqdm(test_data_loader):
        img, label = data['input'].to(device), data['target']
        prediction_outputs, prior_mu, prior_sigma = Prob_UNet.predict_step(img)
        stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
        stacked_samples = torch.sigmoid(stacked_samples)
        prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)
        dice_metric(y_pred=discreter(prediction_heatmap.cpu().unsqueeze(0).unsqueeze(0)), y=discreter(label.unsqueeze(0)))
        dice_score = dice_metric.aggregate().item()
        dice_scores.append(dice_score)
        dice_metric.reset()


#%%

with open(f'results/dice_scores_ProbUnet.json', 'w') as file:
    json.dump(dice_scores, file)

#%%

# import json
# with open('../results/dice_scores_ProbUnet.json', 'r') as file:
#     dice_scores_ProbUnet = json.load(file)

# #%%
# import numpy as np

# #np.array(dice_scores_ProbUnet)
# np.array(dice_scores_ProbUnet).mean()

# #%%
# import matplotlib.pyplot as plt

# # Plot histogram of dice_scores
# plt.hist(dice_scores_ProbUnet, bins=10, edgecolor='black')
# plt.xlabel('Dice Score')
# plt.ylabel('Frequency')
# plt.title('Histogram of Dice Scores')
# plt.show()


#%% Single image dice evaluation

# import os
# from siim_dataset import SIIMDataset

# fold_no = 'testing'
# img_serial = 6    # Good: 6, 92 Bad: 532, 484
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# test_dataset = SIIMDataset(folds=[fold_no], if_test=True)

# image = test_dataset[img_serial]['input']  # shape: [1, 512, 512]
# mask = test_dataset[img_serial]['target']  # shape: [1, 512, 512]

# input_image = image.unsqueeze(0).to(device)
# Prob_UNet.to(device)
# prediction_outputs, prior_mu, prior_sigma = Prob_UNet.predict_step(input_image)
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

#%% Prediction heatmap

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
#            cmap='plasma', aspect='auto',
#            )
# plt.colorbar()
# plt.title(f'Heatmap of Epistemic Uncertainty: {fold_no}_{img_serial}')


#%%

# import json

# # 從 JSON 文件中讀取列表
# with open('../results/dice_scores_Unet.json', 'r') as file:
#     dice_scores_Unet = json.load(file)

# #%%
# import matplotlib.pyplot as plt

# # Draw histogram for dice_scores_ProbUnet
# plt.hist(dice_scores_ProbUnet, bins=10, edgecolor='black', alpha=0.5, label='ProbUNet')

# # Draw histogram for dice_scores
# plt.hist(dice_scores_Unet, bins=10, edgecolor='black', alpha=0.5, label='Original')

# # # Draw histogram for dice_scores_segformer
# # plt.hist(dice_scores_segformer, bins=10, edgecolor='black', alpha=0.5, label='Segformer')


# # Add labels and title
# plt.xlabel('Dice Score')
# plt.ylabel('Frequency')
# plt.title('Histogram of Dice Scores')

# # Add legend
# plt.legend()

# # Show the histogram
# plt.show()


#%%