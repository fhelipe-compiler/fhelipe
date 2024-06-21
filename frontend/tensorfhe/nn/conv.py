# $lic$
# Copyright (C) 2023-2024 by Massachusetts Institute of Technology
#
# This file is part of the Fhelipe compiler.
#
# Fhelipe is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# Fhelipe is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

from typing import Final

import torch

from .. import lib
from ..core import Input, TorchInput
from ..core.utils import is_pow_of_2
from .batch_norm import BatchNorm2d
from .module import FheV, ModuleV, SplitModule, WrapperModule


class Conv2d(WrapperModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        self.in_c: Final = in_channels
        self.out_c: Final = out_channels

        if kernel_size % 2 == 0:
            raise ValueError("Conv2d requires odd kernel_size", kernel_size)
        self.kernel: Final = kernel_size

        if not is_pow_of_2(stride):
            raise ValueError("Conv2d requires power-of-2 stride", stride)
        self.stride: Final = stride

        if bias:
            raise ValueError("Conv2d does not support bias")

        super().__init__(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                bias=False,
                padding=kernel_size // 2,
            )
        )

    @property
    def w_fhe(self) -> Input:
        weights_shape = (
            self.out_c,
            self.in_c,
            self.kernel,
            self.kernel,
        )
        return self.weights("weight", weights_shape)

    def forward_fhe(self, x: FheV) -> FheV:
        x = lib.conv2d(x, self.w_fhe, self.stride)

        return x


class ConvBn2d(SplitModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size, stride, bias=False
        )
        self.bn = BatchNorm2d(out_channels)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv.torch(x)
        x = self.bn.torch(x)
        return x

    def forward_fhe(self, x: FheV) -> FheV:
        bn_mul = self.bn.mul_fhe.reshape(self.bn.mul_fhe.shape + (1,))
        w = self.conv.w_fhe * bn_mul

        x = lib.conv2d(x, w, stride=self.conv.stride)
        return x + self.bn.add_fhe
