#%%

from monai.losses.spatial_mask import MaskedLoss
from torch.nn import BCEWithLogitsLoss
from typing import Any, Optional
import torch

class MaskedBCEWithLogitsLoss(BCEWithLogitsLoss):
    """
    Add an additional `masking` process before `BCEWithLogitsLoss`, accept a binary mask ([0, 1]) indicating a region,
    `input` and `target` will be masked by the region: region with mask `1` will keep the original value,
    region with `0` mask will be converted to `0`. Then feed `input` and `target` to normal `BCEWithLogitsLoss` computation.
    This has the effect of ensuring only the masked region contributes to the loss computation and
    hence gradient calculation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Args follow :py:class:`torch.nn.BCEWithLogitsLoss`.
        """
        super().__init__(*args, **kwargs)
        self.spatial_weighted = MaskedLoss(loss=BCEWithLogitsLoss(*args, **kwargs))

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        return self.spatial_weighted(input=input, target=target, mask=mask)


#%%


