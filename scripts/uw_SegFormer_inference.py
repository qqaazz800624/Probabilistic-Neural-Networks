#%%

from SegFormer_lightningmodule import SegFormerModule
from uw_dataset_segformer import DatasetConfig, TrainingConfig
import os
import torch 
from uw_datamodule_segformer import UW_SegFormerDataModule
from lightning import Trainer
import wandb
from utils_segformer import inference

root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/'
model_weight = 'results/UM_medical_segmentation/version_0/checkpoints/ckpt_010-vloss_0.3404_vf1_0.9195.ckpt'
weight_path = os.path.join(root, model_weight)

model = SegFormerModule(
    model_name=TrainingConfig.MODEL_NAME,
    num_classes=DatasetConfig.NUM_CLASSES,
    init_lr=TrainingConfig.INIT_LR,
    optimizer_name=TrainingConfig.OPTIMIZER_NAME,
    weight_decay=TrainingConfig.WEIGHT_DECAY,
    use_scheduler=TrainingConfig.USE_SCHEDULER,
    scheduler_name=TrainingConfig.SCHEDULER,
    num_epochs=TrainingConfig.NUM_EPOCHS,
) 

model_weight = torch.load(weight_path, map_location="cpu")["state_dict"]
model.load_state_dict(model_weight)
model.eval()

#%%

# Initialize custom data module.
data_module = UW_SegFormerDataModule(
    num_classes=DatasetConfig.NUM_CLASSES,
    img_size=DatasetConfig.IMAGE_SIZE,
    ds_mean=DatasetConfig.MEAN,
    ds_std=DatasetConfig.STD,
    batch_size=TrainingConfig.BATCH_SIZE,
    num_workers=TrainingConfig.NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

data_module.setup()
valid_loader = data_module.val_dataloader()


#%%

trainer = Trainer(
    accelerator="gpu",
    devices=1,        
    enable_checkpointing=False,
    inference_mode=True,
)

# Run evaluation.
results = trainer.validate(model=model, dataloaders=valid_loader)

if os.environ.get("LOCAL_RANK", None) is None:
    wandb.run.summary["best_valid_f1"] = results[0]["valid/f1"]
    wandb.run.summary["best_valid_loss"] = results[0]["valid/loss"]

#%%

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

inference(model, valid_loader, DatasetConfig.IMAGE_SIZE, device=device)




#%%







#%%










#%%







#%%