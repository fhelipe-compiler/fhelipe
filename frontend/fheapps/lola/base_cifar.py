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
from typing import Dict, Generic, TypeVar

import tensorfhe as tfhe
import tensorfhe.app.actions as act
from tensorfhe import Input, Shape, Vector, VectorT
from torchvision.transforms import ToTensor

T = TypeVar("T")


class LolaWeights(act.PopulateSharedTorch):
    path = Path(__file__).parent / "weights" / "cifar.pt"


class LolaTestIn(act.Cifar10TestIn):
    data_transform = ToTensor()


class LolaError(act.ArgMaxOutputMixin, act.Cifar10TestError):
    pass


class BaseLolaCifar(tfhe.App, Generic[T]):
    actions = {*act.base_actions, LolaWeights, LolaTestIn, LolaError}

    @abstractmethod
    def enc(self, x: Input) -> T:
        raise NotImplementedError

    @abstractmethod
    def conv1(self, x: T, w: Input, b: Input) -> T:
        raise NotImplementedError

    @abstractmethod
    def conv2(self, x: T, w: Input, b: Input) -> T:
        raise NotImplementedError

    @abstractmethod
    def fc(self, x: T, w: Input, b: Input) -> T:
        raise NotImplementedError

    @abstractmethod
    def square(self, x: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def dec(self, x: T) -> Vector:
        raise NotImplementedError

    def w(self, name: str, shape: Shape) -> Input:
        return tfhe.public_in(name, shape)

    def save(self, key: str, x: T) -> T:
        self.__out[key] = self.dec(x)
        return x

    def __init__(self, **kwargs) -> None:
        self.__out: Dict[str, Vector] = {}

        x_in = tfhe.secret_in(tfhe.stdin, (3, 32, 32))
        x = self.enc(x_in)

        x = self.conv1(
            x, self.w("conv1.w", (83, 3, 8, 8)), self.w("conv1.b", (83,))
        )
        self.save("conv1", x)

        x = self.square(x)

        x = self.conv2(
            x, self.w("conv2.w", (112, 83, 10, 10)), self.w("conv2.b", (112,))
        )
        self.save("conv2", x)

        x = self.square(x)
        x = self.fc(x, self.w("fc.w", (10, 112, 7, 7)), self.w("fc.b", (10,)))
        self.save(tfhe.stdout, x)

        super().__init__(out=self.__out)
