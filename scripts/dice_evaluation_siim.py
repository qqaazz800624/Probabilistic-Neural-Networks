#%%

import os
import torch
from monai.metrics import DiceMetric
from manafaln.transforms import LoadJSON, OverlayMask, Fill
from manafaln.transforms import ParseXAnnotationSegmentationLabel, Interpolate, Fill, OverlayMask
import matplotlib.pyplot as plt
from instance_seg import InstanceSegmentation
import json
from utils import image_preprocessor, label_preprocessor

def dice_preprocessor(fold_no, 
                      img_serial,
                      version_no = 'version_11',
                      model_name = 'DeepLabV3Plus'):
    json_file = 'datalist.json'
    data_root = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'
    json_path = os.path.join(data_root, json_file)
    generator_input = image_preprocessor(fold_no, img_serial, data_root, datalist=json_path)

    model_weight = f'/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/lightning_logs/{version_no}/checkpoints/best_model.ckpt'  
    model_config = {'name': model_name,
                    'path': 'segmentation_models_pytorch',
                    'args':{
                        'in_channels': 1,
                        'classes': 1,
                        'encoder_name': 'tu-resnest50d',
                        'encoder_weights': 'None'}
                        }
    masks_generator = InstanceSegmentation(model_config=model_config, model_weight=model_weight, instance_only=True)

    generator_output = masks_generator(generator_input)
    prediction = torch.from_numpy(generator_output.cpu().unsqueeze(0).unsqueeze(0).detach().numpy())

    label = label_preprocessor(fold_no, img_serial, data_root, datalist=json_path, keyword='label')
    overlayMasker = OverlayMask(colors=['#7f007f'])
    overlaymask=overlayMasker(image=generator_input, masks=label)
    
    return overlaymask, prediction, label.unsqueeze(0), generator_output, generator_input

#%%

json_file = 'datalist.json'
data_root = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'
json_path = os.path.join(data_root, json_file)
json_file = open(json_path)
img_paths = json.load(json_file)
fold_no='testing'
dice_metric = DiceMetric(include_background=True, reduction='mean')
dice_scores = []

from tqdm import tqdm 
from monai.transforms import AsDiscrete
discreter = AsDiscrete(threshold=0.5)

for img_serial in tqdm(range(len(img_paths[fold_no]))):
    overlaymask, prediction, label, generator_output, original_image = dice_preprocessor(fold_no=fold_no, img_serial=img_serial)
    dice_metric(y_pred=discreter(prediction), y=discreter(label))
    dice_score = dice_metric.aggregate().item()
    dice_scores.append(dice_score)
    dice_metric.reset()


import json

# 將列表保存為 JSON 文件
with open('results/dice_scores.json', 'w') as file:
    json.dump(dice_scores, file)

#%%
    
import json

# 從 JSON 文件中讀取列表
with open('../results/dice_scores.json', 'r') as file:
    dice_scores = json.load(file)

import numpy as np


np.array(dice_scores).mean()

#%%

import matplotlib.pyplot as plt

# Plot histogram of dice_scores
plt.hist(dice_scores, bins=10, edgecolor='black')
plt.xlabel('Dice Score')
plt.ylabel('Frequency')
plt.title('Histogram of Dice Scores')
plt.show()

#%%

model_version_dict = {'DeepLabV3Plus': 'version_11'}

model_name = 'DeepLabV3Plus'
version_no = model_version_dict[model_name]

fold_no = 'testing'
img_serial = 6  # good example: 6, 500  # bad example: 532, 484
overlaymask, prediction, label, generator_output, original_image = dice_preprocessor(fold_no=fold_no, img_serial=img_serial, version_no=version_no, model_name=model_name)


#%% Original image 

plt.imshow(original_image.T, cmap='gray', aspect='auto')

#%% Original image and target mask

plt.imshow(overlaymask.T, cmap='plasma', aspect='auto')

#%% label mask

plt.imshow(label[0].T, cmap='plasma', aspect='auto')

#%% prediction mask 

plt.imshow(generator_output.cpu().T, cmap='plasma', aspect='auto')

#%% computing dice score

# Set include_background to False if you don't want to include the 
#background class (class 0) in the Dice calculation
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
discreter = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction='mean')

# Compute Dice score
dice_metric(y_pred=discreter(prediction), y=discreter(label))
dice_score = dice_metric.aggregate().item()
dice_metric.reset()

print("Dice Score:", dice_score)

#%%


#%%