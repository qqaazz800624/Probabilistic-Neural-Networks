#%%
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from manafaln.transforms import LoadJSON
from monai.transforms import LoadImage, Resize, ScaleIntensity
import os

#%%
def process_segmentation_prediction(
    preds: Tensor, aggregate_fn: Callable = torch.mean, eps: float = 1e-7
) -> dict[str, Tensor]:
    """Process segmentation predictions.

    Applies softmax to logit and computes mean over the samples and entropy.

    Args:
        preds: prediction logits tensor of shape
            [batch_size, num_classes, height, width, num_samples]
        aggregate_fn: function to aggregate over the samples
        eps: small value to prevent log of 0

    Returns:
        dictionary with mean [batch_size, num_classes, height, width]
            and predictive uncertainty [batch_size, height, width]
    """
    # dim=1 is the expected num classes dimension
    agg_logits = aggregate_fn(preds, dim=-1)
    mean = nn.functional.softmax(agg_logits, dim=-1)
    # prevent log of 0 -> nan
    mean.clamp_min_(eps)
    entropy = -(mean * mean.log()).sum(dim=1)

    samples = preds.permute(4, 0, 1, 2, 3)
    
    return {"pred": mean, "pred_uct": entropy, "logits": agg_logits, "samples": samples}

#%%

def image_preprocessor(fold_no, img_serial,
                       data_root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet',
                       datalist = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet/datalist_fold_montgomery.json'): 
    datalist = LoadJSON(json_only=True)(datalist)
    img_path = datalist[fold_no][img_serial]['image']
    image_loader = LoadImage(image_only=True, ensure_channel_first= True)
    resizer = Resize(spatial_size = [512, 512])
    scaler = ScaleIntensity()
    img = image_loader(os.path.join(data_root, img_path))
    img = resizer(img)
    preprocessed_input_image = scaler(img)

    return preprocessed_input_image

#%%

def label_preprocessor(fold_no, img_serial,
                       data_root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet',
                       datalist = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet/datalist_fold_montgomery.json',
                       keyword = 'target'):
    datalist = LoadJSON(json_only=True)(datalist)
    labelfile = os.path.join(data_root, datalist[fold_no][img_serial][keyword])
    image_loader = LoadImage(image_only=True, ensure_channel_first= True)
    resizer = Resize(spatial_size = [512, 512])
    scaler = ScaleIntensity()
    label = image_loader(os.path.join(data_root, labelfile))
    label = resizer(label)
    label = scaler(label)
    return label

#%%

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


#%%