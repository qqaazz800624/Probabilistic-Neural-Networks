from typing import Sequence, Dict, Any

import torch
import numpy as np
from monai.data import MetaTensor
from monai.transforms import MapTransform, RandomizableTransform
from monai.utils.type_conversion import convert_to_numpy, convert_to_tensor
from albumentations import (
    OneOf, Compose, HorizontalFlip, RandomGamma, RandomBrightnessContrast,
    ElasticTransform, GridDistortion, OpticalDistortion, ShiftScaleRotate, Resize
)

class XRayAugs(RandomizableTransform, MapTransform):
    def __init__(
        self,
        img_key: str = "image",
        seg_key: str = "label",
        mask_key: str = "uncertainty_mask",
        img_size: Sequence[int] = (512, 512)
    ) -> None:
        MapTransform.__init__(self, [img_key, seg_key, mask_key], allow_missing_keys=False)
        RandomizableTransform.__init__(self, prob=1.0)

        self.img_key = img_key
        self.seg_key = seg_key
        self.mask_key = mask_key
        self.transform = Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                RandomGamma(),
                RandomBrightnessContrast()
            ], p=0.3),
            OneOf([
                ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    p=0.5
                ),
                GridDistortion(p=1.0),
                OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.0)
            ], p=0.3),
            ShiftScaleRotate(),
            Resize(img_size[0], img_size[1], always_apply=True)
        ])
        self.call_count = 0

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(data)

        if self.img_key not in d:
            raise ValueError(f"Key {self.img_key} is not in the available keys.")
        if self.seg_key not in d:
            raise ValueError(f"Key {self.seg_key} is not in the available keys.")
        if self.mask_key not in d:
            raise ValueError(f"Key {self.mask_key} is not in the available keys.")

        img = convert_to_numpy(d[self.img_key])
        seg = convert_to_numpy(d[self.seg_key])
        mask = convert_to_numpy(d[self.mask_key])

        img = np.transpose(img, axes=(1, 2, 0))
        seg = np.transpose(seg, axes=(1, 2, 0))
        mask = np.transpose(mask, axes=(1, 2, 0))

        # Apply the same transformations to image, seg, and mask
        img_seg = np.concatenate((img, seg, mask), axis=2)
        out = self.transform(image=img_seg, mask=img_seg)

        img_seg = out["image"]
        img, seg, mask = np.split(img_seg, [img.shape[2], img.shape[2] + seg.shape[2]], axis=2)

        img = np.transpose(img, axes=(2, 0, 1))
        img = convert_to_tensor(img)
        if isinstance(d[self.img_key], MetaTensor):
            d[self.img_key] = MetaTensor(img, meta=d[self.img_key].meta)
        else:
            d[self.img_key] = img

        seg = np.transpose(seg, axes=(2, 0, 1))
        seg = convert_to_tensor(seg)
        if isinstance(d[self.seg_key], MetaTensor):
            d[self.seg_key] = MetaTensor(seg, meta=d[self.seg_key].meta)
        else:
            d[self.seg_key] = seg

        mask = np.transpose(mask, axes=(2, 0, 1))
        mask = convert_to_tensor(mask)
        if isinstance(d[self.mask_key], MetaTensor):
            d[self.mask_key] = MetaTensor(mask, meta=d[self.mask_key].meta)
        else:
            d[self.mask_key] = mask

        return d
