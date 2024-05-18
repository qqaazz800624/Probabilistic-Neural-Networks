#%%

from segmentation_models_pytorch import DeepLabV3Plus, Unet
from functools import partial
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from prob_unet import ProbUNet
from utils import image_preprocessor


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 128
model_name = 'DeepLabV3Plus'  # Valid model_name: ['Unet', 'DeepLabV3Plus']
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
version_no = 'version_16'
model_weight = f'/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/lightning_logs/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
Prob_UNet.load_state_dict(model_weight)
Prob_UNet.eval()

#%%

import os

#fold_no = 'fold_4'
fold_no = 'testing'
json_file = 'datalist.json'
data_root = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'
json_path = os.path.join(data_root, json_file)
img_serial = 532

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
preprocessed_input_image = image_preprocessor(fold_no, img_serial, data_root, datalist=json_path)
preprocessed_input_image = preprocessed_input_image.to(device)
Prob_UNet.to(device)
prediction_outputs, prior_mu, prior_sigma = Prob_UNet.predict_step(preprocessed_input_image.unsqueeze(0))


#%%
# from utils import label_preprocessor
# from monai.metrics import DiceMetric
# from monai.transforms import AsDiscrete
# from tqdm import tqdm

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# dice_metric = DiceMetric(include_background=True, reduction='mean')
# discreter = AsDiscrete(threshold=0.5)
# dice_scores = []

# for img_serial in tqdm(range(535)):
#     preprocessed_input_image = image_preprocessor(fold_no, img_serial, data_root, datalist=json_path)
#     preprocessed_input_image = preprocessed_input_image.to(device)
#     Prob_UNet.to(device)
#     prediction_outputs, prior_mu, prior_sigma = Prob_UNet.predict_step(preprocessed_input_image.unsqueeze(0))
#     stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
#     stacked_samples = torch.sigmoid(stacked_samples)
#     prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)
#     prediction_heatmap = prediction_heatmap.cpu()
#     label = label_preprocessor(fold_no, img_serial, data_root, datalist=json_path, keyword='label')
#     dice_metric(y_pred=discreter(prediction_heatmap.unsqueeze(0).unsqueeze(0)), y=discreter(label.unsqueeze(0)))
#     dice_score = dice_metric.aggregate().item()
#     dice_scores.append(dice_score)
#     dice_metric.reset()

# import json

# # 將列表保存為 JSON 文件
# with open('results/dice_scores_ProbUnet.json', 'w') as file:
#     json.dump(dice_scores, file)

#%%
    
import json

# 從 JSON 文件中讀取列表
with open('../results/dice_scores_ProbUnet.json', 'r') as file:
    dice_scores_ProbUnet = json.load(file)

import numpy as np

np.round(np.array(dice_scores_ProbUnet), 3)

#%%

 
import json

# 從 JSON 文件中讀取列表
with open('../results/dice_scores.json', 'r') as file:
    dice_scores = json.load(file)

import numpy as np

np.round(np.array(dice_scores), 3)


#%%

print('dice_scores: ', np.array(dice_scores).mean()) # 0.4276
print('dice_scores_ProbUnet: ', np.array(dice_scores_ProbUnet).mean()) # 0.4803

#%%

counter = 0
for i in range(len(dice_scores)):
    if dice_scores[i] > 0 and dice_scores_ProbUnet[i] == 0:
        counter += 1

counter

#%%

# print('Prior mu: ', prior_mu)
# print('Prior sigma: ', prior_sigma)


#%%

# prediction_outputs.keys() # 'pred', 'pred_uct', 'logits', 'samples'

# import matplotlib.pyplot as plt

# plt.imshow(prediction_outputs['pred'][0, 0].detach().numpy().T, cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Prediction: {fold_no}_{img_serial}')

# #%%

# import matplotlib.pyplot as plt

# plt.imshow(prediction_outputs['pred_uct'][0].detach().numpy().T, cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Uncertainty (Entropy): {fold_no}_{img_serial}')

#%%

stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
stacked_samples = torch.sigmoid(stacked_samples)
uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim=False)

import matplotlib.pyplot as plt

plt.imshow(uncertainty_heatmap.cpu().detach().numpy().T, cmap='plasma', aspect='auto',
           #vmin = 0, vmax = 1
           )
plt.colorbar()
plt.title(f'Heatmap of Epistemic Uncertainty: {fold_no}_{img_serial}')


#%% segmentation samples

# import matplotlib.pyplot as plt

# sample_no = 3
# plt.imshow(stacked_samples[sample_no].detach().numpy().T, cmap='plasma', aspect='auto', 
#            #vmin=-3, vmax=6
#            )
# plt.colorbar()
# plt.title(f'Segmentation samples: {fold_no}_{img_serial}_sample{sample_no}')


#%%

# import matplotlib.pyplot as plt

# for i in range(200):
#     plt.imshow(stacked_samples[i].detach().numpy().T, cmap='plasma', aspect='auto', 
#                #vmin=0, vmax=4
#                )
#     plt.colorbar()
#     plt.title(f'Sample {i}')
#     plt.savefig(f'../results/image_samples/{fold_no}_{img_serial}_sample{i}.png')
#     plt.close()

# import torch

# torch.equal(stacked_samples[2], stacked_samples[3])


#%% Prediction heatmap (Revised)

prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)

import matplotlib.pyplot as plt

plt.imshow(prediction_heatmap.cpu().detach().numpy().T, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Prediction heatmap: {fold_no}_{img_serial}')


#%%

import os
from utils import label_preprocessor

label = label_preprocessor(fold_no, img_serial, data_root, datalist=json_path, keyword='label')
#prediction_heatmap.unsqueeze(0).unsqueeze(0).shape, label.unsqueeze(0).shape


from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
discreter = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction='mean')

dice_metric(y_pred=discreter(prediction_heatmap.cpu().unsqueeze(0).unsqueeze(0)), y=discreter(label.unsqueeze(0)))


#%%