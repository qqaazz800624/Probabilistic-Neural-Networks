#%%


from SegFormer_lightningmodule import SegFormerModule
from siim_dataset_segformer import DatasetConfig, TrainingConfig, SIIMDataset
import torch 
import os
import torch.nn.functional as F
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import json

os.environ['WANDB_NOTEBOOK_NAME'] = 'dice_evaluation_siim'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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

root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/'
model_weight = 'results/SIIM_pneumothorax_segmentation/version_0/checkpoints/ckpt_035-vloss_0.4050_vf1_0.7711.ckpt'
weight_path = os.path.join(root, model_weight)

model_weight = torch.load(weight_path, map_location="cpu")["state_dict"]
model.load_state_dict(model_weight)
model.eval()
model.to(device)

#%%

dice_scores_segformer = []

fold_no = 'testing'
dataset_test = SIIMDataset(folds=[fold_no])
discreter = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction='mean')


for img_serial in tqdm(range(len(dataset_test))):
    image, mask = dataset_test[img_serial]
    image = image.unsqueeze(0).to(device)
    logits = model(image)
    predictions = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu()
    dice_metric(y_pred=discreter(predictions), y=discreter(mask.unsqueeze(0)))
    dice_score = dice_metric.aggregate().item()
    dice_scores_segformer.append(dice_score)
    dice_metric.reset()


# 將列表保存為 JSON 文件
with open('results/dice_scores_segformer.json', 'w') as file:
    json.dump(dice_scores_segformer, file)

#%%

import json

# 從 JSON 文件中讀取列表
with open('../results/dice_scores_segformer.json', 'r') as file:
    dice_scores_segformer = json.load(file)


#%%

# 從 JSON 文件中讀取列表
with open('../results/dice_scores.json', 'r') as file:
    dice_scores = json.load(file)

np.array(dice_scores)

#%%

import numpy as np

np.array(dice_scores_segformer).mean()

#%%

np.array(dice_scores_segformer)



#%%

# img_serial = 6  # good example: 6, 500  # bad example: 532, 484
# image, mask = dataset_test[img_serial]
# image = image.unsqueeze(0)
# logits = model(image)

# #%%

# import torch.nn.functional as F

# predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)


# #%%


# from monai.metrics import DiceMetric
# from monai.transforms import AsDiscrete
# discreter = AsDiscrete(threshold=0.5)
# dice_metric = DiceMetric(include_background=True, reduction='mean')

# # Compute Dice score
# dice_metric(y_pred=discreter(predictions), y=discreter(mask.unsqueeze(0)))
# dice_score = dice_metric.aggregate().item()
# dice_metric.reset()

# print("Dice Score:", dice_score)


# #%%

# import matplotlib.pyplot as plt

# plt.imshow(predictions.squeeze(0).detach().numpy(), 
#            cmap='plasma',
#            aspect='auto')


#%%





#%%







#%%







#%%