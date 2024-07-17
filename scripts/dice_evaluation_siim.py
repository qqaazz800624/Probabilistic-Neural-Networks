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


# #%%
import os, json, torch
from tqdm import tqdm
from siim_datamodule import SIIMDataModule
#from siim_datamodule_balancedSampler import SIIMDataModule
from segmentation_models_pytorch import Unet, DeepLabV3Plus
from custom.mednext import mednext_base, mednext_large
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from unet_lightningmodule import UNetModule

model_name = 'mednext'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

data_module = SIIMDataModule(batch_size_test=1, num_workers_test=2)
test_data_loader = data_module.test_dataloader()

if model_name == 'Unet':
    model = Unet(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
    model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
    model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
    for k in list(model_weight.keys()):
        k_new = k.replace(
            "model.", "", 1
        )  # e.g. "model.conv.weight" => conv.weight"
        model_weight[k_new] = model_weight.pop(k)
    model.load_state_dict(model_weight)
    
elif model_name == 'DeepLabV3Plus':
    model = DeepLabV3Plus(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
    model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_78/checkpoints/best_model.ckpt'
    model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
    for k in list(model_weight.keys()):
        k_new = k.replace(
            "model.", "", 1
        )  # e.g. "model.conv.weight" => conv.weight"
        model_weight[k_new] = model_weight.pop(k)
    model.load_state_dict(model_weight)

elif model_name == 'mednext':
    #model = mednext_base(in_channels=1, out_channels=1, spatial_dims=2, use_grad_checkpoint=True)
    model = mednext_large(in_channels=1, out_channels=1, spatial_dims=2, use_grad_checkpoint=True)
    model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_90/checkpoints/best_model.ckpt'
    model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
    for k in list(model_weight.keys()):
        k_new = k.replace(
            "model.", "", 1
        )  # e.g. "model.conv.weight" => conv.weight"
        model_weight[k_new] = model_weight.pop(k)
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

with open(f'results/dice_scores_MedNeXt.json', 'w') as file:
    json.dump(dice_scores, file)

#%%
# with open(f'results/dice_scores_{model_name}.json', 'w') as file:
#     json.dump(dice_scores, file)


#%% Single image dice evaluation

import os, json, torch
from tqdm import tqdm
from siim_datamodule import SIIMDataModule
from siim_dataset import SIIMDataset
from segmentation_models_pytorch import Unet, DeepLabV3Plus
from custom.mednext import mednext_base, mednext_large

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from manafaln.transforms import OverlayMask
from utils import image_preprocessor, label_preprocessor


model_name = 'mednext'
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
ckpt_path = 'results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

if model_name == 'Unet':
    model = Unet(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
    model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
    model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
    for k in list(model_weight.keys()):
        k_new = k.replace(
            "model.", "", 1
        )  # e.g. "model.conv.weight" => conv.weight"
        model_weight[k_new] = model_weight.pop(k)
    model.load_state_dict(model_weight)
    
elif model_name == 'DeepLabV3Plus':
    model = DeepLabV3Plus(in_channels=1, classes=1, encoder_name = 'tu-resnest50d', encoder_weights = 'imagenet')
    model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_78/checkpoints/best_model.ckpt'
    model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
    for k in list(model_weight.keys()):
        k_new = k.replace(
            "model.", "", 1
        )  # e.g. "model.conv.weight" => conv.weight"
        model_weight[k_new] = model_weight.pop(k)
    model.load_state_dict(model_weight)

elif model_name == 'mednext':
    #model = mednext_base(in_channels=1, out_channels=1, spatial_dims=2, use_grad_checkpoint=True)
    model = mednext_large(in_channels=1, out_channels=1, spatial_dims=2, use_grad_checkpoint=True)
    model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_90/checkpoints/best_model.ckpt'
    model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
    for k in list(model_weight.keys()):
        k_new = k.replace(
            "model.", "", 1
        )  # e.g. "model.conv.weight" => conv.weight"
        model_weight[k_new] = model_weight.pop(k)
    model.load_state_dict(model_weight)

# model_weight = torch.load(os.path.join(root_dir, ckpt_path), map_location="cpu")["state_dict"]
# for k in list(model_weight.keys()):
#     k_new = k.replace("model.", "", 1)
#     model_weight[k_new] = model_weight.pop(k)
# model.load_state_dict(model_weight)
model.eval()
model.to(device)

#%%
fold_no = 'testing'
# Good: 6, 92, 522, 292 Bad: 532, 484, 168, 163, 207, 212
# large mask: 92, 417, 492, 339, 132, 302
# large-medium mask: 338, 325, 377
# medium mask: 107, 136
# medium-small mask: 29, 412
# small mask: 128, 184

img_serial = 163
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