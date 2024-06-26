from typing import Sequence

import torch
import numpy as np
import monai
from monai.data import MetaTensor
from monai.transforms import MapTransform, RandomizableTransform
from monai.utils.type_conversion import (
    convert_to_numpy,
    convert_to_tensor
)
from albumentations import (
    OneOf,
    Compose,
    HorizontalFlip,
    RandomGamma,
    RandomBrightnessContrast,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    ShiftScaleRotate,
    Resize
)

class XRayAugs(RandomizableTransform, MapTransform):
    def __init__(
        self,
        img_key: str = "image",
        seg_key: str = "label",
        mask_key: str = "mask",
        img_size: Sequence[int] = (512, 512)
    ) -> None:
        MapTransform.__init__(self, [img_key, seg_key, mask_key], False)
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

    def __call__(self, data) -> dict:
        d: dict = dict(data)

        img = convert_to_numpy(d[self.img_key])
        seg = convert_to_numpy(d[self.seg_key])
        mask = convert_to_numpy(d[self.mask_key])

        img = np.transpose(img, axes=(1, 2, 0))
        seg = np.transpose(seg, axes=(1, 2, 0))
        mask = np.transpose(mask, axes=(1, 2, 0))
        out = self.transform(image=img, mask=seg, uncertainty_mask=mask)

        img = np.transpose(out["image"], axes=(2, 0, 1))
        img = convert_to_tensor(img)
        if isinstance(d[self.img_key], MetaTensor):
            d[self.img_key] = MetaTensor(img, meta=d[self.img_key].meta)
        else:
            d[self.img_key] = img

        seg = np.transpose(out["mask"], axes=(2, 0, 1))
        seg = convert_to_tensor(seg)
        if isinstance(d[self.seg_key], MetaTensor):
            d[self.seg_key] = MetaTensor(seg, meta=d[self.seg_key].meta)
        else:
            d[self.seg_key] = seg

        mask = np.transpose(out["uncertainty_mask"], axes=(2, 0, 1))
        mask = convert_to_tensor(mask)
        if isinstance(d[self.mask_key], MetaTensor):
            d[self.mask_key] = MetaTensor(mask, meta=d[self.mask_key].meta)
        else:
            d[self.mask_key] = mask

        return d
