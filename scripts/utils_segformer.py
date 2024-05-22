#%%

import torch.nn.functional as F


#%%


def dice_coef_loss(predictions, ground_truths, num_classes=2, dims=(1, 2), smooth=1e-8):
    """Smooth Dice coefficient + Cross-entropy loss function."""
 
    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)
    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)
 
    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
    summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)
 
    dice = (2.0 * intersection + smooth) / (summation + smooth)
    dice_mean = dice.mean()
 
    CE = F.cross_entropy(predictions, ground_truths)
 
    return (1.0 - dice_mean) + CE



#%%

from transformers import SegformerForSemanticSegmentation
from uw_dataset_segformer import TrainingConfig, DatasetConfig

def get_model(*, model_name, num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


#%%


id2color = {
    0: (0, 0, 0),    # background pixel
    1: (0, 0, 255),  # Stomach
    2: (0, 255, 0),  # Small Bowel
    3: (255, 0, 0),  # large Bowel
}

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def num_to_rgb(num_arr, color_map=id2color):
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))
 
    for k in color_map.keys():
        output[single_layer == k] = color_map[k]
 
    # return a floating point array in range [0.0, 1.0]
    return np.float32(output) / 255.0


def image_overlay(image, segmented_image):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.
 
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    return np.clip(image, 0.0, 1.0)


def display_image_and_mask(*, images, masks, color_map=id2color):
    title = ["GT Image", "Color Mask", "Overlayed Mask"]
 
    for idx in range(images.shape[0]):
        image = images[idx]
        grayscale_gt_mask = masks[idx]
 
        fig = plt.figure(figsize=(15, 4))
 
        # Create RGB segmentation map from grayscale segmentation map.
        rgb_gt_mask = num_to_rgb(grayscale_gt_mask, color_map=color_map)
 
        # Create the overlayed image.
        overlayed_image = image_overlay(image, rgb_gt_mask)
 
        plt.subplot(1, 3, 1)
        plt.title(title[0])
        plt.imshow(image)
        plt.axis("off")
 
        plt.subplot(1, 3, 2)
        plt.title(title[1])
        plt.imshow(rgb_gt_mask)
        plt.axis("off")
 
        plt.imshow(rgb_gt_mask)
        plt.subplot(1, 3, 3)
        plt.title(title[2])
        plt.imshow(overlayed_image)
        plt.axis("off")
 
        plt.tight_layout()
        plt.show()
 
    return

def denormalize(tensors, *, mean, std):
    for c in range(DatasetConfig.CHANNELS):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
 
    return torch.clamp(tensors, min=0.0, max=1.0)

#%%