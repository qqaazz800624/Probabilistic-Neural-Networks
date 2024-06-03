#%%

# import os
# import torch
# from monai.metrics import DiceMetric
# from manafaln.transforms import LoadJSON, OverlayMask, Fill
# from manafaln.transforms import ParseXAnnotationSegmentationLabel, Interpolate, Fill, OverlayMask
# import matplotlib.pyplot as plt
# from instance_seg import InstanceSegmentation
# import json
# from utils import image_preprocessor, label_preprocessor


# def dice_preprocessor(fold_no, 
#                       img_serial,
#                       version_no = 'version_1',         # version_no = ['version_1', 'version_0']
#                       model_name = 'DeepLabV3Plus'):    # model_name = ['DeepLabV3Plus', 'Unet']
#     json_file = 'datalist.json'
#     data_root = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'
#     json_path = os.path.join(data_root, json_file)
#     generator_input = image_preprocessor(fold_no, img_serial, data_root, datalist=json_path)

#     model_weight = f'/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/lightning_logs/{version_no}/checkpoints/best_model.ckpt'  
#     model_config = {'name': model_name,
#                     'path': 'segmentation_models_pytorch',
#                     'args':{
#                         'in_channels': 1,
#                         'classes': 1,
#                         'encoder_name': 'tu-resnest50d',
#                         'encoder_weights': 'None'}
#                         }
#     masks_generator = InstanceSegmentation(model_config=model_config, model_weight=model_weight, instance_only=True)

#     generator_output = masks_generator(generator_input)
#     prediction = torch.from_numpy(generator_output.cpu().unsqueeze(0).unsqueeze(0).detach().numpy())

#     label = label_preprocessor(fold_no, img_serial, data_root, datalist=json_path, keyword='label')
#     overlayMasker = OverlayMask(colors=['#7f007f'])
#     overlaymask=overlayMasker(image=generator_input, masks=label)
    
#     return overlaymask, prediction, label.unsqueeze(0), generator_output, generator_input


#%%
import os, json, torch
from tqdm import tqdm
from siim_datamodule import SIIMDataModule
#from siim_datamodule_balancedSampler import SIIMDataModule
from segmentation_models_pytorch import Unet, DeepLabV3Plus
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from unet_lightningmodule import UNetModule

model_name = 'Unet'
model_version_dict = {'Unet': 'version_14',
                      'DeepLabV3Plus': 'version_16'}
#version_no = model_version_dict[model_name]
version_no = 'version_22'
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
ckpt_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_module = SIIMDataModule(batch_size_test=1, num_workers_test=2)
test_data_loader = data_module.test_dataloader()

model = UNetModule(loss_fn_name='DiceCELoss')
# if model_name == 'Unet':
#     model = Unet(in_channels=1, 
#                  classes=1, 
#                  encoder_name = 'tu-resnest50d', 
#                  encoder_weights = 'imagenet')
    
# elif model_name == 'DeepLabV3Plus':
#     model = DeepLabV3Plus(in_channels=1, 
#                           classes=1, 
#                           encoder_name = 'tu-resnest50d', 
#                           encoder_weights = 'imagenet')

model_weight = torch.load(os.path.join(root_dir, ckpt_path), map_location="cpu")["state_dict"]
# for k in list(model_weight.keys()):
#     k_new = k.replace("model.", "", 1)
#     model_weight[k_new] = model_weight.pop(k)
model.load_state_dict(model_weight)
model.eval()
model.to(device)

dice_metric = DiceMetric(include_background=True, reduction='none', ignore_empty=False)
discreter = AsDiscrete(threshold=0.5)
dice_scores = []

with torch.no_grad():
    for data in tqdm(test_data_loader):
        img, label = data['input'].to(device), data['target'].to(device)
        prediction = model(img)
        dice_metric(y_pred=discreter(prediction), y=discreter(label))
        dice_score = dice_metric.aggregate().item()
        dice_scores.append(dice_score)
        dice_metric.reset()

print('Mean Dice Score:', sum(dice_scores) / len(dice_scores))
#%%

with open(f'results/dice_scores_Unet_DiceFocalLoss.json', 'w') as file:
    json.dump(dice_scores, file)

#%%
# with open(f'results/dice_scores_{model_name}.json', 'w') as file:
#     json.dump(dice_scores, file)

#%%

# number of total testing images: 2411
# number of labeled images: 535
# number of unlabeled images: 1876

# import numpy as np

# nan_count = np.sum(np.isnan(dice_scores))
# print("Number of NaN values:", nan_count)


#%%

# import json
# import numpy as np

# # 從 JSON 文件中讀取列表
# with open('../results/dice_scores_Unet.json', 'r') as file:
#     dice_scores_Unet = json.load(file)

# #np.array(dice_scores_Unet)
# #np.round(np.array(dice_scores_Unet), 4)
# np.mean(dice_scores_Unet)

#%%

# import json
# import numpy as np

# # 從 JSON 文件中讀取列表
# with open('../results/dice_scores_DeepLabV3Plus.json', 'r') as file:
#     dice_scores_DeepLabV3Plus = json.load(file)

# #np.array(dice_scores_DeepLabV3Plus)
# np.mean(dice_scores_DeepLabV3Plus)


#%%

# import matplotlib.pyplot as plt

# # Plot histogram of dice_scores_Unet
# plt.hist(dice_scores_Unet, bins=10, edgecolor='black')
# plt.xlabel('Dice Score')
# plt.ylabel('Frequency')
# plt.title('Histogram of Dice Scores')
# plt.show()

# #%%

# # Plot histogram of dice_scores_DeepLabV3Plus
# plt.hist(dice_scores_DeepLabV3Plus, bins=10, edgecolor='black')
# plt.xlabel('Dice Score')
# plt.ylabel('Frequency')
# plt.title('Histogram of Dice Scores')
# plt.show()

# #%%

# plt.hist(dice_scores_Unet, bins=10, edgecolor='black', alpha=0.5, label='Unet')
# plt.hist(dice_scores_DeepLabV3Plus, bins=10, edgecolor='black', alpha=0.5, label='DeepLabV3Plus')
# plt.xlabel('Dice Score')
# plt.ylabel('Frequency')
# plt.title('Histogram of Dice Scores')
# plt.legend()
# plt.show()


#%% Single image dice evaluation

import os, json, torch
from tqdm import tqdm
from siim_datamodule import SIIMDataModule
from siim_dataset import SIIMDataset
from segmentation_models_pytorch import Unet, DeepLabV3Plus
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from manafaln.transforms import OverlayMask
from utils import image_preprocessor, label_preprocessor


model_name = 'Unet'
model_version_dict = {'Unet': 'version_0',
                      'DeepLabV3Plus': 'version_1'}
version_no = model_version_dict[model_name]
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
#ckpt_path = f'results/lightning_logs/{version_no}/checkpoints/best_model.ckpt'
ckpt_path = 'results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
    
model_weight = torch.load(os.path.join(root_dir, ckpt_path), map_location="cpu")["state_dict"]
for k in list(model_weight.keys()):
    k_new = k.replace("model.", "", 1)
    model_weight[k_new] = model_weight.pop(k)
model.load_state_dict(model_weight)
model.eval()
model.to(device)

#%%
fold_no = 'testing'
# Good: 6, 92, 522 Bad: 532, 484, 168
# large mask: 92, 417, 492, 339, 132, 302
# large-medium mask: 338, 325
# medium mask: 107, 136
# medium-small mask: 29, 412
# small mask: 128, 184

img_serial = 132
test_dataset = SIIMDataset(folds=[fold_no], if_test=True)
image = test_dataset[img_serial]['input']
mask = test_dataset[img_serial]['target']

input_image = image.unsqueeze(0).to(device)
prediction = model(input_image)
prediction = torch.sigmoid(prediction)

discreter = AsDiscrete(threshold=0.5)

json_file = 'datalist.json'
data_root = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'
json_path = os.path.join(data_root, json_file)

overlay_image = image_preprocessor(fold_no, img_serial, data_root, datalist=json_path)
overlay_label = label_preprocessor(fold_no, img_serial, data_root, datalist=json_path, keyword='label')
overlayMasker = OverlayMask(colors=['#7f007f'])
overlaymask=overlayMasker(image = overlay_image, masks = overlay_label)

#%%

from tqdm import tqdm 
import torch, json

fold_no = 'testing'
test_dataset = SIIMDataset(folds=[fold_no], if_test=True)
mask_size = []

for img_serial in tqdm(range(535)):
    mask = test_dataset[img_serial]['target']
    mask_size.append(mask.sum())

mask_size = torch.tensor(mask_size)
sorted, indices =  torch.sort(mask_size, descending=True)

with open('../results/mask_size.json', 'w') as file:
    json.dump(indices.tolist(), file)

#%%
import json

with open('../results/mask_size.json', 'r') as file:
    indices = json.load(file)
#%%

indices[15]

#%%

import matplotlib.pyplot as plt
# Plot overlay mask
plt.imshow(overlaymask.T, cmap='gray', aspect='auto')

#%%
import matplotlib.pyplot as plt

# Plot original image
plt.imshow(image[0].T, cmap='gray', aspect='auto')

#%%

# Plot target mask
plt.imshow(mask[0].T, cmap='plasma', aspect='auto')

#%%
# Plot prediction mask

plt.imshow(discreter(prediction[0][0].detach().cpu().T), cmap='plasma', aspect='auto')

#%% computing dice score

# Set include_background to False if you don't want to include the 
#background class (class 0) in the Dice calculation
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
discreter = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction='none')

# Compute Dice score
dice_metric(y_pred=discreter(prediction.cpu()), y=discreter(mask.unsqueeze(0)))
dice_score = dice_metric.aggregate().item()
dice_metric.reset()

print("Dice Score:", dice_score)

#%% self-defined dice function

def dice(prediction, mask):
    intersection = (prediction[0, 0].cpu() * mask[0]).sum()
    total_elements = prediction[0, 0].sum() + mask[0].sum()
    dice = (2 * intersection) / total_elements
    return dice.item()

dice_score = dice(prediction, mask)
print("Dice Score:", dice_score)

#%%