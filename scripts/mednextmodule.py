#%%

import torch
from lightning import LightningModule
from custom.mednext import mednext_base, mednext_large, mednext_medium
from segmentation_models_pytorch.losses import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class MedNeXtModule(LightningModule):
    def __init__(self,
                 in_channels: int = 1,
                 spatial_dims: int = 2,
                 out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.model = mednext_large(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            spatial_dims=self.spatial_dims,
            use_grad_checkpoint=True
        )
        self.loss_fn = DiceLoss(mode='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch['input'], batch['target']
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch['input'], batch['target']
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch['input'], batch['target']
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3.0e-4, weight_decay=1e-5)
        
        # Instantiate the scheduler
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1)
        
        # Define the scheduler configuration
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',  # Scheduler step is called every epoch
            'frequency': 1,  # Scheduler step is called once per epoch
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}

#%%