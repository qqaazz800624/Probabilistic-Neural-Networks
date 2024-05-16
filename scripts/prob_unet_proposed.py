#%%


# Copyright 2019 Stefan Knegt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
# Changes from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/blob/master/probabilistic_unet.py: # noqa: E501
# - adapt ProbUnet implementation to lightning training framework
# - make Unet flexible to be any segmentation model

"""Probabilistic U-Net."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.distributions import kl

from lightning_uq_box.uq_methods import BaseModule

#from lightning_uq_box.models.prob_unet import AxisAlignedConvGaussian, Fcomb
from lightning_uq_box.uq_methods.utils import default_segmentation_metrics
from utils import process_segmentation_prediction
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss
from axisalignedconvgaussian import AxisAlignedConvGaussian, Fcomb


class ProbUNet(BaseModule):
    """Probabilistic U-Net.

    If you use this code, please cite the following paper:

    * https://arxiv.org/abs/1806.05034
    """

    valid_tasks = ["multiclass", "binary"]

    def __init__(
        self,
        model: nn.Module,
        model_name: str = 'Unet',
        loss_fn: str = 'DiceLoss',
        batch_size_train: int = 6,
        latent_dim: int = 6,
        num_filters: List[int] = [32, 64, 128, 192],
        num_convs_per_block: int = 3,
        num_convs_fcomb: int = 4,
        fcomb_filter_size: int = 32,
        beta: float = 10.0,
        gamma: float = -1,
        lambd: float = 10,
        num_samples: int = 100,
        max_epochs: int = 32,
        task: str = "multiclass",
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialize a new instance of ProbUNet.

        Args:
            model: Unet model
            latent_dim: latent dimension
            num_filters: number of filters per block in AxisAlignedConvGaussian
            num_convs_per_block: num of convs per block in AxisAlignedConvGaussian
            num_convs_fcomb: number of convolutions in fcomb
            fcomb_filter_size: filter size for the fcomb network
            beta: beta parameter
            num_samples: number of latent samples to use during prediction
            task: task type, either "multiclass" or "binary"
            optimizer: optimizer
            lr_scheduler: learning rate scheduler
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])

        self.batch_size_train = batch_size_train
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.latent_dim = latent_dim
        self.num_convs_fcomb = num_convs_fcomb
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd
        self.num_filters = num_filters
        self.fcomb_filter_size = fcomb_filter_size
        self.num_convs_per_block = num_convs_per_block
        self.num_samples = num_samples

        assert task in self.valid_tasks, f"Task must be one of {self.valid_tasks}."
        self.task = task

        self.model = model
        self.num_input_channels = self.num_input_features
        self.num_classes = self.num_outputs
        self.prior = AxisAlignedConvGaussian(
            self.num_input_channels,
            self.num_filters,
            self.num_convs_per_block,
            self.latent_dim,
            posterior=False,
        )
        self.posterior = AxisAlignedConvGaussian(
            self.num_input_channels,
            self.num_filters,
            self.num_convs_per_block,
            self.latent_dim,
            posterior=True,
        )
        self.fcomb_input_channels = self.num_classes + self.latent_dim
        self.fcomb = Fcomb(
            self.fcomb_input_channels,
            self.fcomb_filter_size,
            self.num_classes,
            self.num_convs_fcomb,
            initializers={"w": "orthogonal", "b": "normal"},
            use_tile=True,
        )

        if loss_fn == 'BCEWithLogitsLoss':
            self.criterion = BCEWithLogitsLoss(reduction="none")  # original setting, version_3
        elif loss_fn == 'DiceLoss':
            self.criterion = DiceLoss(mode='binary')            # experimental setting

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.setup_task()

    def setup_task(self) -> None:
        """Set up the task."""
        self.train_metrics = default_segmentation_metrics(
            prefix="train", num_classes=self.num_classes, task=self.task
        )
        self.val_metrics = default_segmentation_metrics(
            prefix="val", num_classes=self.num_classes, task=self.task
        )
        self.test_metrics = default_segmentation_metrics(
            prefix="test", num_classes=self.num_classes, task=self.task
        )

    def compute_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute the evidence lower bound (ELBO) of the log-likelihood of P(Y|X).

        Args:
            seg_mask: segmentation target mask

        Returns:
            A dictionary containing the total loss,
            reconstruction loss, KL loss, and the reconstruction
        """
        
        img, seg_mask = batch[self.input_key], batch[self.target_key]
        # check dimensions, add channel dimension to seg_mask under assumption
        # that it is a binary mask
        # print('Part 1')
        # print('The shape of seg_mask: ', seg_mask.shape)
        # print('The shape of img: ', img.shape)
        # The shape of seg_mask: [batch_size, height, weight] --> [6, 512, 512]
        if len(seg_mask.shape) == 3:
            seg_mask_target = seg_mask.long()
            seg_mask_target = F.one_hot(seg_mask_target, num_classes=self.num_classes)
            seg_mask_target = seg_mask_target.permute(
                0, 3, 1, 2
            ).float()  # move class dim to the channel dim

            # channel dimension for concatenation
            seg_mask = seg_mask.unsqueeze(1)
            # The shape of seg_mask: [batch_size, num_class, height, weight] --> [6, 1, 512, 512]
            # The shape of seg_mask_target: [batch_size, num_class, height, weight] --> [6, 1, 512, 512]
        else:
            seg_mask_target = seg_mask

        # print('Part 2')
        # print('The shape of seg_mask: ', seg_mask.shape)
        # print('The shape of seg_mask_target: ', seg_mask_target.shape)

        self.posterior_latent_space, self.posterior_mu, self.posterior_sigma = self.posterior.forward(img, seg_mask)
        self.prior_latent_space, self.prior_mu, self.prior_sigma = self.prior.forward(img)
        print('prior net mu: ', self.prior_mu)
        print('prior net sigma: ', self.prior_sigma)
        print('posterior net mu: ', self.posterior_mu)
        print('posterior net sigma: ', self.posterior_sigma)

        self.unet_features = self.model.forward(img)

        z_posterior = self.posterior_latent_space.rsample()

        kl_loss = torch.mean(self.kl_divergence(analytic=True, z_posterior=z_posterior))
        print('KL divergence: ', kl_loss)

        reconstruction = self.reconstruct(
            use_posterior_mean=False, z_posterior=z_posterior
        )

        # print('Part 3')
        # print('The shape of seg_mask_target: ', seg_mask_target.shape)
        # print('The shape of reconstruction: ', reconstruction.shape)
        rec_loss_y2 = self.criterion(self.unet_features, seg_mask_target)
        rec_loss_y2_sum = torch.sum(rec_loss_y2)
        rec_loss_y1 = self.criterion(reconstruction, seg_mask_target)
        rec_loss_y1_sum = torch.sum(rec_loss_y1)
        penalty = rec_loss_y1_sum - rec_loss_y2_sum - self.gamma

        loss = self.beta*kl_loss + rec_loss_y1_sum + rec_loss_y2_sum + self.lambd*penalty 

        # rec_loss = self.criterion(reconstruction, seg_mask_target)
        # rec_loss_sum = torch.sum(rec_loss)
        # rec_loss_mean = torch.mean(rec_loss)

        # #loss = -(rec_loss_sum + self.beta * kl_loss) # original version
        # loss = (rec_loss_sum + self.beta * kl_loss) # version 2

        return {
            "loss": loss,
            "rec_loss_y2_sum": rec_loss_y2_sum,
            "rec_loss_y1_sum": rec_loss_y1_sum,
            "kl_loss": kl_loss,
            "penalty": penalty,
            "reconstruction": reconstruction
        }

    def kl_divergence(
        self, analytic: bool = True, z_posterior: Optional[Tensor] = None
    ) -> Tensor:
        """Compute the KL divergence between the posterior and prior KL(Q||P).

        Args:
            analytic: calculate KL analytically or via sampling from the posterior
            z_posterior: if we use sampling to approximate KL we can sample here or
                supply a sample

        Returns:
            The KL divergence
        """
        if analytic:
            # TODO this should not be necessary anymore to add this to torch source
            # see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(
                self.posterior_latent_space, self.prior_latent_space
            )
        else:
            if z_posterior is None:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def reconstruct(
        self, use_posterior_mean: bool = False, z_posterior: Optional[Tensor] = None
    ) -> Tensor:
        """Reconstruct a segmentation from a posterior sample.

        Decoding a posterior sample and UNet feature map

        Args:
            use_posterior_mean: use posterior_mean instead of sampling z_q
            z_posterior: use a provided sample or sample from posterior latent space

        Returns:
            The reconstructed segmentation
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if z_posterior is None:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def sample(self, testing: bool = False) -> Tensor:
        """Sample a segmentation via reconstructing from a prior sample.

        Args:
            testing: whether to sample from the prior or use the mean
        """
        if testing is False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here.
            # For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        loss_dict = self.compute_loss(batch)

        self.log("train_loss", loss_dict["loss"], on_epoch=True, prog_bar=True, logger=True)
        self.log("train_rec_loss_y1_sum", loss_dict["rec_loss_y1_sum"], on_epoch=True, logger=True)
        self.log("train_rec_loss_y2_sum", loss_dict["rec_loss_y2_sum"], on_epoch=True, logger=True)
        self.log("train_kl_loss", loss_dict["kl_loss"], on_epoch=True, logger=True)
        self.log("train_penalty", loss_dict["penalty"], on_epoch=True, logger=True, prog_bar=True)

        self.train_metrics(loss_dict["reconstruction"], batch[self.target_key])

        # self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_rec_loss_sum", loss_dict["rec_loss_sum"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_rec_loss_mean", loss_dict["rec_loss_mean"])
        # self.log("train_kl_loss", loss_dict["kl_loss"])

        # compute metrics with reconstruction
        #self.train_metrics(loss_dict["reconstruction"], batch[self.target_key])

        # return loss to optimize
        return loss_dict["loss"]

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the validation loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        loss_dict = self.compute_loss(batch)

        self.log("val_loss", loss_dict["loss"])
        self.log("val_rec_loss_y1_sum", loss_dict["rec_loss_y1_sum"])
        self.log("val_rec_loss_y2_sum", loss_dict["rec_loss_y2_sum"])
        self.log("val_kl_loss", loss_dict["kl_loss"])
        # compute metrics with reconstruction
        self.val_metrics(loss_dict["reconstruction"], batch[self.target_key])

        return loss_dict["loss"]

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Tensor]:
        """Compute and return the test loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        Returns:
            test prediction dict
        """
        preds = self.predict_step(batch[self.input_key])

        # compute metrics with sampled reconstruction
        self.test_metrics(preds["logits"], batch[self.target_key])

        preds = self.add_aux_data_to_dict(preds, batch)

        return preds

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, Tensor]:
        """Compute and return the prediction.

        Args:
            X: the input image
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        Returns:
            prediction dict
        """
        # this internally computes the latent space and unet features
        self.prior_latent_space, self.prior_mu, self.prior_sigma = self.prior.forward(X)
        self.unet_features = self.model.forward(X)
        # which can then be used to sample a segmentation
        
        samples = torch.stack(
            [self.sample(testing=True) for _ in range(self.num_samples)], dim=-1
        )  # shape: (batch_size, num_classes, height, width, num_samples)

        return process_segmentation_prediction(samples), self.prior_mu, self.prior_sigma


    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


#%%




#%%




#%%