from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.checkpoint import checkpoint as grad_ckpt

__all__ = [
    "MedNeXtBlock",
    "MedNeXtDownBlock",
    "MedNeXtUpBlock",
    "OutBlock",
    "MedNeXt",
    "mednext_small",
    "mednext_base",
    "mednext_medium",
    "mednext_large"
]

class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        res_block: bool = True
    ):
        super().__init__()

        if spatial_dims == 2:
            Conv = nn.Conv2d
        elif spatial_dims == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError(f"MedNeXtBlock only support 2D or 3D inputs.")

        self.res_block = res_block

        layers = OrderedDict()
        layers["conv1"] = Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels
        )
        layers["norm"] = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )
        layers["conv2"] = Conv(
            in_channels=in_channels,
            out_channels=in_channels * expand_ratio,
            kernel_size=1,
            stride=1,
            padding=0
        )
        layers["act"] = nn.GELU()
        layers["conv3"] = Conv(
            in_channels=in_channels * expand_ratio,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0

        )
        self.layers = nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        s = self.layers(x)
        if self.res_block:
            return s + x
        return s

class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        res_block: bool = True
    ):
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            expand_ratio,
            kernel_size,
            res_block=False
        )

        # Get convolution type and replace it
        Conv = type(self.layers.conv1)
        self.layers.conv1 = Conv(
            in_channels,
            in_channels,
            kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels
        )

        if res_block:
            self.res_conv = Conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2
            )
        else:
            self.res_conv = None

    def forward(self, x: Tensor) -> Tensor:
        s = super().forward(x)

        if self.res_conv is not None:
            r = self.res_conv(x)
            s = s + r

        return s

class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        res_block: bool = True
    ):
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            expand_ratio,
            kernel_size,
            res_block=False
        )

        # Note: MedNeXtBlock should ensure `spatial_dims` is either 2 or 3
        if spatial_dims == 2:
            self.pad_sizes = (1, 0, 1, 0)
        else:
            self.pad_sizes = (1, 0, 1, 0, 1, 0)
        Conv = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d

        self.layers.conv1 = Conv(
            in_channels,
            in_channels,
            kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels
        )

        if res_block:
            self.res_conv = Conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2
            )
        else:
            self.res_conv = None

    def forward(self, x: Tensor) -> Tensor:
        s = super().forward(x)
        s = nn.functional.pad(s, self.pad_sizes)

        if self.res_conv is not None:
            r = self.res_conv(x)
            r = nn.functional.pad(r, self.pad_sizes)
            s = s + r

        return s

class OutBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()

        if spatial_dims == 2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1)
        elif spatial_dims == 3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> torch.Tensor:
        return self.conv(x)

class MedNeXt(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        filters: int,
        num_blocks: Sequence[int],
        expand_ratio: Union[int, Sequence[int]],
        res_block: bool = True,
        deep_supervision: bool = False,
        use_grad_checkpoint: bool = False
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.res_block = res_block
        self.deep_supervision = deep_supervision
        self.use_grad_checkpoint = use_grad_checkpoint

        if not len(num_blocks) % 2:
            raise ValueError(
                "The length of `blocks` must be 2n+1, where n is the number of downsamples."
            )

        self.num_blocks = num_blocks
        self.depth = len(self.num_blocks) // 2 

        if isinstance(expand_ratio, int):
            self.expand_ratio = [expand_ratio] * len(num_blocks)
        elif isinstance(expand_ratio, Sequence) and (len(expand_ratio) != len(self.num_blocks)):
            raise ValueError("The length of `num_blocks` and `expand_ratio` must be the same.")
        else:
            self.expand_ratio = list(expand_ratio)

        # Input layer
        if self.spatial_dims == 2:
            self.stem = nn.Conv2d(in_channels, self.filters, kernel_size=1)
        elif self.spatial_dims == 3:
            self.stem = nn.Conv3d(in_channels, self.filters, kernel_size=1)
        else:
            raise ValueError(
                f"Invalid spatial dimensions ({self.spatial_dims}), only 2 or 3 is allowed."
            )

        # Construct encoder
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(self.depth):
            in_chs = self.filters * (2 ** i)
            out_chs = in_chs * 2

            self.enc_blocks.append(
                nn.Sequential(*[
                    MedNeXtBlock(
                        spatial_dims=spatial_dims,
                        in_channels=in_chs,
                        out_channels=in_chs,
                        expand_ratio=expand_ratio[i],
                        kernel_size=kernel_size,
                        res_block=res_block
                    )
                    for j in range(num_blocks[i])
                ])
            )
            self.down_blocks.append(
                MedNeXtDownBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_chs,
                    out_channels=out_chs,
                    expand_ratio=expand_ratio[i + 1],
                    kernel_size=kernel_size,
                    res_block=res_block
            ))

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                spatial_dims=spatial_dims,
                in_channels=self.filters * (2 ** self.depth),
                out_channels=self.filters * (2 ** self.depth),
                kernel_size=kernel_size,
                expand_ratio=expand_ratio[self.depth],
                res_block=res_block
            )
            for i in range(num_blocks[self.depth])
        ])

        # Construct decoder
        self.dec_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.depth)):
            idx = len(expand_ratio) - 1 - i
            in_chs = self.filters * (2 ** (i + 1))
            out_chs = in_chs // 2

            self.up_blocks.append(
                MedNeXtUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_chs,
                    out_channels=out_chs,
                    expand_ratio=expand_ratio[idx],
                    kernel_size=kernel_size,
                    res_block=res_block
                )
            )
            self.dec_blocks.append(
                nn.Sequential(*[
                    MedNeXtBlock(
                        spatial_dims=spatial_dims,
                        in_channels=out_chs,
                        out_channels=out_chs,
                        expand_ratio=expand_ratio[idx],
                        kernel_size=kernel_size,
                        res_block=res_block
                    )
                    for j in range(num_blocks[idx])
                ])
            )

        # Output layers
        self.out = OutBlock(spatial_dims, self.filters, out_channels)
        if deep_supervision:
            self.ds_out_blocks = nn.ModuleList()
            for i in range(self.depth, 0, -1):
                self.ds_out_blocks.append(
                    OutBlock(spatial_dims, self.filters * (2 ** i), out_channels)
                )

        # Dummy tensor with require_grad to fix checkpointing bug
        # self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def encode(self, x: Tensor):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.down_blocks):
            s = enc(x)
            x = down(s)
            skips.append(s)
        x = self.bottleneck(x)
        return x, skips

    def encode_with_ckpt(self, x: Tensor):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.down_blocks):
            s = x
            for layer in enc:
                s = grad_ckpt(layer, s, use_reentrant=False)
            x = grad_ckpt(down, s, use_reentrant=False)
            skips.append(s)
        x = grad_ckpt(self.bottleneck, x, use_reentrant=False)
        return x, skips

    def decode(self, x: Tensor, skips: List[Tensor]):
        for up, dec in zip(self.up_blocks, self.dec_blocks):
            x = up(x) + skips.pop()
            x = dec(x)
        x = self.out(x)
        return x

    def decode_with_ckpt(self, x: Tensor, skips: List[Tensor]):
        for up, dec in zip(self.up_blocks, self.dec_blocks):
            x = grad_ckpt(up, x, use_reentrant=False) + skips.pop()
            for layer in dec:
                x = grad_ckpt(layer, x, use_reentrant=False)
        x = grad_ckpt(self.out, x, use_reentrant=False)
        return x

    def decode_with_ds(self, x: Tensor, skips: List[Tensor]) -> Tensor:
        ds_heads = []
        for ds_out, up, dec in zip(self.ds_out_blocks, self.up_blocks, self.dec_blocks):
            ds_heads.append(ds_out(x))
            x = up(x) + skips.pop()
            x = dec(x)
        x = self.out(x)

        # Collect deep supervision outputs and transform to MONAI format
        out = [x]
        for ds_head in reversed(ds_heads):
            out.append(interpolate(ds_head, x.shape[2:]))
        x = torch.stack(out, dim=1)

        return x

    def decode_with_ds_ckpt(self, x: Tensor, skips: List[Tensor]) -> Tensor:
        ds_heads = []
        for ds_out, up, dec in zip(self.ds_out_blocks, self.up_blocks, self.dec_blocks):
            ds_heads.append(grad_ckpt(ds_out, x, use_reentrant=False))
            x = grad_ckpt(up, x, use_reentrant=False) + skips.pop()
            x = grad_ckpt(dec, x, use_reentrant=False)
        x = grad_ckpt(self.out, x, use_reentrant=False)

        # Collect deep supervision outputs and transform to MONAI format
        out = [x]
        for ds_head in reversed(ds_heads):
            out.append(interpolate(ds_head, x.shape[2:]))
        x = torch.stack(out, dim=1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.training and (self.deep_supervision or self.use_grad_checkpoint):
            if self.use_grad_checkpoint:
                x, skips = self.encode_with_ckpt(x)
                if self.deep_supervision:
                    x = self.decode_with_ds_ckpt(x, skips)
                else:
                    x = self.decode_with_ckpt(x, skips)
            else:
                x, skips = self.encode(x)
                x = self.decode_with_ds(x, skips)
        else:
            x, skips = self.encode(x)
            x = self.decode(x, skips)
        return x


def mednext_small(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    filters: int = 32,
    deep_supervision: bool = False,
    use_grad_checkpoint: bool = False
) -> MedNeXt:
    model = MedNeXt(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        filters=filters,
        num_blocks=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        expand_ratio=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        res_block=True,
        deep_supervision=deep_supervision,
        use_grad_checkpoint=use_grad_checkpoint
    )
    return model

def mednext_base(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    filters: int = 32,
    deep_supervision: bool = False,
    use_grad_checkpoint: bool = False
) -> MedNeXt:
    model = MedNeXt(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        filters=filters,
        num_blocks=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        expand_ratio=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        res_block=True,
        deep_supervision=deep_supervision,
        use_grad_checkpoint=use_grad_checkpoint
    )
    return model

def mednext_medium(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    filters: int = 32,
    deep_supervision: bool = False,
    use_grad_checkpoint: bool = False
) -> MedNeXt:
    model = MedNeXt(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        filters=filters,
        num_blocks=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        expand_ratio=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        res_block=True,
        deep_supervision=deep_supervision,
        use_grad_checkpoint=use_grad_checkpoint
    )
    return model

def mednext_large(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    filters: int = 32,
    deep_supervision: bool = False,
    use_grad_checkpoint: bool = False
) -> MedNeXt:
    model = MedNeXt(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        filters=filters,
        num_blocks=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        expand_ratio=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        res_block=True,
        deep_supervision=deep_supervision,
        use_grad_checkpoint=use_grad_checkpoint
    )
    return model

