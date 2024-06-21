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

from ..core import Input, Shape
from .module import FheV, WrapperModule


class BatchNorm2d(WrapperModule):
    def __init__(
        self,
        c: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        self.__eps: Final = eps
        self.__c: Final = c

        super().__init__(
            torch.nn.BatchNorm2d(c, eps=eps, momentum=momentum, affine=affine)
        )

    def __weights(self, key: str) -> Input:
        return self.weights(key, (self.__c,)).reshape((self.__c, 1, 1))

    @property
    def mul_fhe(self) -> Input:
        weight = self.__weights("weight")
        var = self.__weights("running_var")

        return weight / (var + self.__eps).sqrt()

    @property
    def add_fhe(self) -> Input:
        bias = self.__weights("bias")
        mean = self.__weights("running_mean")

        return bias - mean * self.mul_fhe

    def forward_fhe(self, x: FheV) -> FheV:
        return x * self.mul_fhe + self.add_fhe
