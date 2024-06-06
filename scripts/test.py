#%%

from segmentation_models_pytorch import Unet
from functools import partial
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from prob_unet import ProbUNet
import os
from siim_datamodule import SIIMDataModule
from tqdm import tqdm

#%%

beta = 10
latent_dim = 6
max_epochs = 128
model_name = 'Unet'
batch_size_train = 16
num_samples = 30
loss_fn = 'BCEWithLogitsLoss'

unet = Unet(in_channels=1,
            classes=1,
            encoder_name='tu-resnest50d',
            encoder_weights='imagenet')

Prob_UNet = ProbUNet(
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
    num_samples=num_samples
    )

#version_no = model_version_dict[model_name]
version_no = 'version_24'
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
Prob_UNet.load_state_dict(model_weight)
Prob_UNet.eval()

#%%

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

data_module = SIIMDataModule(batch_size_test=16, num_workers_test=2)
test_loader = data_module.test_dataloader()

Prob_UNet.to(device)

counter = 0
with torch.no_grad():
    for data in tqdm(test_loader):
        img, label = data['input'].to(device), data['target']
        prediction_outputs, prior_mu, prior_sigma = Prob_UNet.predict_step(img)
        counter += 1
        if counter == 1:
            break


#%%


stacked_samples = torch.sigmoid(prediction_outputs['samples'])
uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim = False).cpu()

#%%
uncertainty_heatmap

#%%

import torch
from tqdm import tqdm

# Assuming uncertainty_heatmap is a tensor of shape [batch_size, 1, height, width]
# And mask is a tensor of shape [batch_size, 1, height, width]

batch_size = uncertainty_heatmap.shape[0]
mask_uncertainty = torch.zeros_like(label)

for i in tqdm(range(batch_size)):
    heatmap = uncertainty_heatmap[i, 0]  # Select the ith heatmap
    mask_i = label[i, 0]  # Select the ith mask
    
    # Compute the 97.5th quantile for the ith heatmap
    quantile = torch.kthvalue(heatmap.flatten(), int(0.975 * heatmap.numel())).values.item()
    
    # Apply the threshold to create the mask for the ith element
    mask_uncertainty[i, 0] = torch.where(heatmap > quantile, mask_i, torch.zeros_like(mask_i))

# Now mask_uncertainty contains the masked values for each batch element

#%%

mask_uncertainty.shape


#%%

import matplotlib.pyplot as plt

plt.imshow(uncertainty_heatmap[1].squeeze(0).cpu().detach().numpy().T, 
           cmap='plasma', aspect='auto')
plt.colorbar()
plt.title(f'Heatmap of Epistemic Uncertainty')


#%%

import torch

# Assume these are the two [16, 1, 512, 512] tensors
tensor1 = torch.randn(16, 1, 512, 512)
tensor2 = torch.randn(16, 1, 512, 512)

# Perform element-wise multiplication
result = tensor1 * tensor2

# Print the shape to confirm it remains [16, 1, 512, 512]
print(result.shape)  # Should output torch.Size([16, 1, 512, 512])


#%%

result[1]




#%%






#%%






#%%