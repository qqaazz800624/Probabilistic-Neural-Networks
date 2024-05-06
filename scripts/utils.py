#%%
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable


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

