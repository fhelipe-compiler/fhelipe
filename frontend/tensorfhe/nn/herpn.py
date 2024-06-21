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

from math import pi, sqrt
from typing import Final

import torch

from ..core import as_input
from .batch_norm import BatchNorm2d
from .module import FheV, ModuleV, SplitModule


class HerPN2d(SplitModule):
    f: Final = (1 / sqrt(2 * pi), 1 / 2, 1 / sqrt(4 * pi))

    # TODO is this at all necessary?
    @staticmethod
    def h0(x: ModuleV) -> ModuleV:
        return x.from_const(1).broadcast_to(x.shape)

    @staticmethod
    def h1(x: ModuleV) -> ModuleV:
        return x

    @staticmethod
    def h2(x: ModuleV) -> ModuleV:
        return x * x - 1

    def __init__(self, c: int):
        super().__init__()

        self.bn0: Final = BatchNorm2d(c=c)
        self.bn1: Final = BatchNorm2d(c=c)
        self.bn2: Final = BatchNorm2d(c=c)

        self.bn: Final = (self.bn0, self.bn1, self.bn2)
        self.h: Final = (self.h0, self.h1, self.h2)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        x_in = as_input(x)

        y = sum(bn(f * h(x_in)) for bn, f, h in zip(self.bn, self.f, self.h))
        assert not isinstance(y, int)

        return y.tensor

    def forward_fhe(self, x: FheV) -> FheV:
        y = sum(
            # Equivalent to `bn(f * h(x))`
            (bn.mul_fhe * f) * h(x) + bn.add_fhe
            for bn, f, h in zip(self.bn, self.f, self.h)
        )
        assert not isinstance(y, int)

        return y
