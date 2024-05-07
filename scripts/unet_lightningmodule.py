#%%

import torch
from lightning import LightningModule
from torchmetrics import JaccardIndex
from segmentation_models_pytorch import DeepLabV3Plus, Unet
from segmentation_models_pytorch.losses import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss

class UNetModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Unet(
            in_channels=1,
            classes=1,
            encoder_name='tu-resnest50d',
            encoder_weights='imagenet'
        )
        self.loss_fn = DiceLoss(mode='binary')
        #self.loss_fn = BCEWithLogitsLoss(reduction="none")

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
