#%%

import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from transformers import SegformerForSemanticSegmentation
from uw_dataset_segformer import DatasetConfig, InferenceConfig
import wandb, os

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


def get_model(*, pretrained_model_name, num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
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

@torch.inference_mode()
def inference(model, loader, img_size, device="cpu"):
    num_batches_to_process = InferenceConfig.NUM_BATCHES
 
    for idx, (batch_img, batch_mask) in enumerate(loader):
        predictions = model(batch_img.to(device))
 
        pred_all = predictions.argmax(dim=1).cpu().numpy()
 
        batch_img = denormalize(batch_img.cpu(), mean=DatasetConfig.MEAN, std=DatasetConfig.STD)
        batch_img = batch_img.permute(0, 2, 3, 1).numpy()
 
        if idx == num_batches_to_process:
            break
 
        for i in range(0, len(batch_img)):
            fig = plt.figure(figsize=(20, 8))
 
            # Display the original image.
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(batch_img[i])
            ax1.title.set_text("Actual frame")
            plt.axis("off")
 
            # Display the ground truth mask.
            true_mask_rgb = num_to_rgb(batch_mask[i], color_map=id2color)
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.set_title("Ground truth labels")
            ax2.imshow(true_mask_rgb)
            plt.axis("off")
 
            # Display the predicted segmentation mask.
            pred_mask_rgb = num_to_rgb(pred_all[i], color_map=id2color)
            ax3 = fig.add_subplot(1, 4, 3)
            ax3.set_title("Predicted labels")
            ax3.imshow(pred_mask_rgb)
            plt.axis("off")
 
            # Display the predicted segmentation mask overlayed on the original image.
            overlayed_image = image_overlay(batch_img[i], pred_mask_rgb)
            ax4 = fig.add_subplot(1, 4, 4)
            ax4.set_title("Overlayed image")
            ax4.imshow(overlayed_image)
            plt.axis("off")
            plt.show()
             
            # Upload predictions to WandB.
            images = wandb.Image(fig, caption=f"Prediction Sample {idx}_{i}")
             
            if os.environ.get("LOCAL_RANK", None) is None:
                wandb.log({"Predictions": images})


#%%








#%%