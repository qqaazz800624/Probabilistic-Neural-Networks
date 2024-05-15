import torch
from monai.transforms import MapTransform
from typing import Union, List, Dict

class GrayscaleToRGBd(MapTransform):
    def __init__(self, keys: Union[str, List[str]] = 'image'):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
        """
        super().__init__(keys)
        self.keys = keys if isinstance(keys, list) else [keys]

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if img.shape[0] == 1:  # Single channel image
                img = img.repeat(3, 1, 1)  # Repeat the single channel to create an RGB image
            d[key] = img
        return d