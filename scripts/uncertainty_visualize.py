#%%

from segmentation_models_pytorch import DeepLabV3Plus, Unet
from functools import partial
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from prob_unet import ProbUNet
from utils import image_preprocessor


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 32
model_name = 'DeepLabV3Plus'  # Valid model_name: ['Unet', 'DeepLabV3Plus']
latent_dim = 6
beta = 10
batch_size_train = 12
# model_version_dict = {'Unet': 'version_6',
#                       'DeepLabV3Plus': 'version_7'}

# ============ Inference setting ============= #
num_samples = 200
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
version_no = 'version_2'
model_weight = f'/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/lightning_logs/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
Prob_UNet.load_state_dict(model_weight)
Prob_UNet.eval()


#%%

fold_no = 'fold_4'
img_serial = 0
preprocessed_input_image = image_preprocessor(fold_no, img_serial)
prediction_outputs = Prob_UNet.predict_step(preprocessed_input_image.unsqueeze(0))

#%%

#prediction_outputs.keys() # 'pred', 'pred_uct', 'logits', 'samples'

import matplotlib.pyplot as plt

plt.imshow(prediction_outputs['pred'][0, 0].detach().numpy().T, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Prediction: {fold_no}_{img_serial}')

#%%

import matplotlib.pyplot as plt

plt.imshow(prediction_outputs['pred_uct'][0].detach().numpy().T, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Uncertainty (Entropy): {fold_no}_{img_serial}')

#%%

stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim=False)

import matplotlib.pyplot as plt

plt.imshow(uncertainty_heatmap.detach().numpy().T, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Heatmap of Epistemic Uncertainty: {fold_no}_{img_serial}')


#%% segmentation samples

import matplotlib.pyplot as plt

plt.imshow(stacked_samples[139].detach().numpy().T, cmap='plasma', aspect='auto', 
           vmin=0, vmax=4)
plt.colorbar()
plt.title(f'Segmentation samples: {fold_no}_{img_serial}')


#%%

import torch

torch.equal(stacked_samples[2], stacked_samples[3])


#%% Prediction heatmap (Revised)

prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)

import matplotlib.pyplot as plt

plt.imshow(prediction_heatmap.detach().numpy().T, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Prediction heatmap: {fold_no}_{img_serial}')


#%%




#%%