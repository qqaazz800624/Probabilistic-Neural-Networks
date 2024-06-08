#%%

import torch
import os
from monai.transforms import Transform
from segmentation_models_pytorch import Unet
from prob_unet_self_correction import ProbUNet_Proposed
from functools import partial
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

#%%

class UncertaintyHeatmap(Transform):
    def __init__(self,
                 beta: int = 10,
                 latent_dim: int = 6,
                 max_epochs: int = 64,
                 model_name: str = 'Unet',
                 batch_size_train: int = 16,
                 loss_fn: str = 'BCEWithLogitsLoss',
                 num_samples: int = 30,
                 version: str = 'version_27'
                 ):
        self.beta = beta
        self.latent_dim = latent_dim
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.batch_size_train = batch_size_train
        self.loss_fn = loss_fn
        self.num_samples = num_samples
        self.version = version
        
        self.unet = Unet(in_channels=1, 
                        classes=1, 
                        encoder_name = 'tu-resnest50d', 
                        encoder_weights = 'imagenet')
        
        self.prob_unet = ProbUNet_Proposed(
            model=self.unet,
            optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
            task='binary',
            lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
            beta = self.beta,
            latent_dim = self.latent_dim,
            max_epochs=self.max_epochs,
            model_name=self.model_name,
            batch_size_train=self.batch_size_train,
            loss_fn=self.loss_fn,
            num_samples=self.num_samples
        )

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
        self.weight_path = f'results/SIIM_pneumothorax_segmentation/{self.version}/checkpoints/best_model.ckpt'
        self.model_wight = torch.load(os.path.join(self.root_dir, self.weight_path), map_location="cpu")["state_dict"]
        self.prob_unet.load_state_dict(self.model_wight)
        self.prob_unet.to(self.device)
        self.prob_unet.requires_grad_(False)
        self.prob_unet.eval()

    def __call__(self, image, mask):
        prediction_outputs, prior_mu_, prior_sigma_ = self.prob_unet.predict_step(image)
        stacked_samples = torch.sigmoid(prediction_outputs['samples'])
        uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim = False)
        batch_size = uncertainty_heatmap.shape[0]
        mask_uncertainty = torch.zeros_like(uncertainty_heatmap)
        for i in range(batch_size):
            heatmap = uncertainty_heatmap[i, 0]
            mask_i = mask[i, 0]
            quantile = torch.kthvalue(heatmap.flatten(), int(0.975 * heatmap.numel())).values.item()
            mask_uncertainty[i, 0] = torch.where(heatmap > quantile, mask_i, torch.zeros_like(mask_i))
        return mask_uncertainty

#%%
