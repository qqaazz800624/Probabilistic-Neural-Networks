#%%

import os, torch
from segmentation_models_pytorch import Unet
from prob_unet_first import ProbUNet_First
from functools import partial
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

loss_fn = 'BCEWithLogitsLoss'
beta = 10
latent_dim = 6
max_epochs = 128
model_name = 'Unet' 
batch_size_train = 16

unet = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')

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
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First.load_state_dict(model_weight)
ProbUnet_First.eval()
ProbUnet_First.requires_grad_(False)

#%%

from siim_datamodule import SIIMDataModule
from tqdm import tqdm   

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

data_module = SIIMDataModule(batch_size_train=16,
                             batch_size_val=1,
                             batch_size_test=1, 
                             num_workers_test=2)

train_data_loader = data_module.train_dataloader()
val_data_loader = data_module.val_dataloader()
test_data_loader = data_module.test_dataloader()

ProbUnet_First.to(device)

all_uncertainty_masks = []

with torch.no_grad():
    for data in tqdm(train_data_loader):
        img, label = data['input'].to(device), data['target']
        prediction_outputs, prior_mu, prior_sigma = ProbUnet_First.predict_step(img)
        stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
        stacked_samples = torch.sigmoid(stacked_samples)
        uncertainty_heatmap = stacked_samples.var(dim=0, keepdim=False)
        batch_size = uncertainty_heatmap.shape[0]
        mask_uncertainty = torch.zeros_like(uncertainty_heatmap)
        for i in range(batch_size):
            heatmap = uncertainty_heatmap[i, 0]
            quantile = torch.quantile(heatmap.flatten(), 0.975).item()
            mask_uncertainty[i, 0] = torch.where(heatmap > quantile, 1, 0)
        all_uncertainty_masks.append(mask_uncertainty)

all_uncertainty_masks = torch.cat(all_uncertainty_masks, dim=0)
#%%

torch.save(all_uncertainty_masks, os.path.join(root_dir, f'results/SIIM_pneumothorax_segmentation/uncertainty_masks_train.pt'))




#%%






#%%






#%%