#%%

from uw_dataset_segformer import DatasetConfig, InferenceConfig
from uw_datamodule_segformer import UW_SegFormerDataModule
import torch.nn.functional as F
import torch 

#%%

# dm = UW_SegFormerDataModule(
#     num_classes=DatasetConfig.NUM_CLASSES,
#     img_size=DatasetConfig.IMAGE_SIZE,
#     ds_mean=DatasetConfig.MEAN,
#     ds_std=DatasetConfig.STD,
#     batch_size=InferenceConfig.BATCH_SIZE,
#     num_workers=2,
#     shuffle_validation=True,
# )

# # Donwload dataset.
# dm.prepare_data()
 
# # Create training & validation dataset.
# dm.setup()
 
# train_loader, valid_loader = dm.train_dataloader(), dm.val_dataloader()


#%% Display a batch of images and masks.

# from utils_segformer import display_image_and_mask, denormalize

# for batch_images, batch_masks in valid_loader:
 
#     batch_images = denormalize(batch_images, mean=DatasetConfig.MEAN, std=DatasetConfig.STD).permute(0, 2, 3, 1).numpy()
#     batch_masks  = batch_masks.numpy()
 
#     print("batch_images shape:", batch_images.shape)
#     print("batch_masks shape: ", batch_masks.shape)
     
#     display_image_and_mask(images=batch_images, masks=batch_masks)
 
#     break

#%%

from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score
from torch import optim
from utils_segformer import dice_coef_loss, get_model

class UW_SegFormerModule(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        init_lr: float = 0.001,
        optimizer_name: str = "Adam",
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        scheduler_name: str = "multistep_lr",
        num_epochs: int = 100,
    ):
        super().__init__()
 
        # Save the arguments as hyperparameters.
        self.save_hyperparameters()
 
        # Loading model using the function defined above.
        self.model = get_model(model_name=self.hparams.model_name, num_classes=self.hparams.num_classes)
 
        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average="macro")
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average="macro")
 
    def forward(self, data):
        outputs = self.model(pixel_values=data, return_dict=True)
        upsampled_logits = F.interpolate(outputs["logits"], size=data.shape[-2:], mode="bilinear", align_corners=False)
        return upsampled_logits
     
    def training_step(self, batch, *args, **kwargs):
        data, target = batch
        logits = self(data)
 
        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = dice_coef_loss(logits, target, num_classes=self.hparams.num_classes)
         
        self.mean_train_loss(loss, weight=data.shape[0])
        self.mean_train_f1(logits.detach(), target)
 
        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=False)
        self.log("train/batch_f1", self.mean_train_f1, prog_bar=True, logger=False)
        return loss
 
    def on_train_epoch_end(self):
        # Computing and logging the training mean loss & mean f1.
        self.log("train/loss", self.mean_train_loss, prog_bar=True)
        self.log("train/f1", self.mean_train_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)
 
    def validation_step(self, batch, *args, **kwargs):
        data, target = batch
        logits = self(data)
         
        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = dice_coef_loss(logits, target, num_classes=self.hparams.num_classes)
 
        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_f1.update(logits, target)
 
    def on_validation_epoch_end(self):
         
        # Computing and logging the validation mean loss & mean f1.
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True)
        self.log("valid/f1", self.mean_valid_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)
 
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )
 
        LR = self.hparams.init_lr
        WD = self.hparams.weight_decay
 
        if self.hparams.optimizer_name in ("AdamW", "Adam"):
            optimizer = getattr(torch.optim, self.hparams.optimizer_name)(self.model.parameters(), lr=LR, 
                                                                          weight_decay=WD, amsgrad=True)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=LR, weight_decay=WD)
 
        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.trainer.max_epochs // 2,], gamma=0.1)
 
            # The lr_scheduler_config is a dictionary that contains the scheduler
            # and its associated configuration.
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch", "name": "multi_step_lr"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
 
        else:
            return optimizer



#%%


