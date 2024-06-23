from typing import Sized
from random import choices, shuffle

import torch
from torch.utils.data import Sampler
from monai.data import Dataset, DataLoader


class BalancedDataSampler(Sampler):
    def __init__(self, data_source: Sized):
        super().__init__(data_source)
        self.data_source = data_source

        self.neg_samples = [i for i, case in enumerate(self.data_source.samples) if "empty" in case["label"]]
        self.pos_samples = [i for i in range(len(self.data_source.samples)) if i not in self.neg_samples]

        self.num_pos_samples = len(self.pos_samples)
        self.num_neg_samples = min(self.num_pos_samples, len(self.neg_samples))
        self.num_samples = self.num_pos_samples + self.num_neg_samples

    def __iter__(self):
        pos_subset = choices(self.pos_samples, k=self.num_pos_samples)
        neg_subset = choices(self.neg_samples, k=self.num_neg_samples)
        subset = pos_subset + neg_subset
        shuffle(subset)

        yield from subset

    def __len__(self):
        return self.num_samples

# class BalancedDataSampler(Sampler):
#     def __init__(self, data_source: Sized):
#         super().__init__()

#         data = data_source.data
#         self.neg_samples = [i for i, case in enumerate(data) if "empty" in case["label"]]
#         self.pos_samples = [i for i in range(len(data)) if i not in self.neg_samples]

#         self.num_pos_samples = len(self.pos_samples)
#         self.num_neg_samples = min(self.num_pos_samples ,len(self.neg_samples))
#         self.num_samples = self.num_pos_samples + self.num_neg_samples

#     def __iter__(self):
#         pos_subset = choices(self.pos_samples, k=self.num_pos_samples)
#         neg_subset = choices(self.neg_samples, k=self.num_neg_samples)
#         subset = pos_subset + neg_subset
#         shuffle(subset)

#         yield from subset

#     def __len__(self):
#         return self.num_samples


def balanced_data_loader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0
) -> DataLoader:
    data_sampler = BalancedDataSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data_sampler,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=False
    )
