#%%

import random
from typing import Sequence, Callable, Dict, Any

import torch
from monai.transforms import MapTransform, Randomizable, apply_transform

class OneOf(Randomizable, MapTransform):
    def __init__(
        self,
        keys: Sequence[str],
        transforms: Sequence[Callable],
        weights: Sequence[float] = None
    ) -> None:
        super().__init__(keys)
        self.transforms = transforms
        self.weights = weights if weights is not None else [1.0 / len(transforms)] * len(transforms)

    def randomize(self) -> None:
        self.choice = random.choices(range(len(self.transforms)), weights=self.weights, k=1)[0]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.randomize()
        transform = self.transforms[self.choice]
        return apply_transform(transform, data)



#%%

import random
from typing import Sequence, Callable, Dict, Any
from monai.transforms import MapTransform, Randomizable, apply_transform

class OneOfd(Randomizable, MapTransform):
    def __init__(
        self,
        keys: Sequence[str],
        transforms: Sequence[Callable],
        weights: Sequence[float] = None,
        prob: float = 1.0,
    ) -> None:
        super().__init__(keys)
        self.transforms = transforms
        self.weights = weights if weights is not None else [1.0 / len(transforms)] * len(transforms)
        self.prob = prob

    def randomize(self) -> None:
        self._do_transform = self.R.random() < self.prob
        if self._do_transform:
            self.choice = random.choices(range(len(self.transforms)), weights=self.weights, k=1)[0]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.randomize()
        if not self._do_transform:
            return data
        transform = self.transforms[self.choice]
        return apply_transform(transform, data)


#%%