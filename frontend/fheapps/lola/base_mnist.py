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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generic

import tensorfhe as tfhe
import tensorfhe.app.actions as act
import torch
from tensorfhe import Input, Shape, VectorT


class LolaWeights(act.PopulateSharedTorch):
    path = Path(__file__).parent / "weights" / "mnist.pt"


class LolaError(act.ArgMaxOutputMixin, act.MnistTestError):
    pass


class BaseLolaMnist(tfhe.App, Generic[VectorT]):
    actions = {
        *act.base_actions,
        act.MnistTestIn,
        LolaWeights,
        LolaError,
    }

    @abstractmethod
    def enc(self, x: Input) -> VectorT:
        raise NotImplementedError

    @abstractmethod
    def conv(self, x: VectorT, w: Input, b: Input) -> VectorT:
        raise NotImplementedError

    @abstractmethod
    def fc1(self, x: VectorT, w: Input, b: Input) -> VectorT:
        raise NotImplementedError

    @abstractmethod
    def fc2(self, x: VectorT, w: Input, b: Input) -> VectorT:
        raise NotImplementedError

    def square(self, x: VectorT) -> VectorT:
        return x * x

    def w(self, name: str, shape: Shape) -> Input:
        return tfhe.public_in(name, shape)

    def save(self, key: str, x: VectorT) -> None:
        self.__out[key] = x

    def __init__(self, **kwargs) -> None:
        self.__out: Dict[str, VectorT] = {}

        x_in = tfhe.secret_in(tfhe.stdin, (1, 28, 28))
        x = self.enc(x_in)

        x = self.conv(x, self.w("conv.w", (5, 1, 5, 5)), self.w("conv.b", (5,)))
        self.save("conv", x)

        x = self.square(x)
        self.save("square1", x)

        x = self.fc1(
            x, self.w("fc1.w", (100, 5, 13, 13)), self.w("fc1.b", (100,))
        )
        self.save("fc1", x)

        x = self.square(x)
        self.save("square2", x)

        x = self.fc2(x, self.w("fc2.w", (10, 100)), self.w("fc2.b", (10,)))
        self.save("fc2", x)

        self.save(tfhe.stdout, x)

        super().__init__(out=self.__out)
