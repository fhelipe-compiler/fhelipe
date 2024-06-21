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

import torch

from .. import lib
from .module import FheV, SplitModule


class GlobalAvgPool2d(SplitModule):
    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        kernel = x.size()[-2:]
        x = torch.nn.functional.avg_pool2d(x, kernel)
        x = torch.flatten(x, start_dim=-3)
        return x

    def forward_fhe(self, x: FheV) -> FheV:
        return lib.global_avg_pool2d(x)


class AvgPool2d(SplitModule):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(self.kernel_size - 1) // 2,
        )

    def forward_fhe(self, x: FheV) -> FheV:
        return lib.avg_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride
        )
