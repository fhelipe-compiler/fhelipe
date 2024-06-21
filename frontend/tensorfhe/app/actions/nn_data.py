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

import os
import sys
from abc import ABC, abstractmethod
from typing import Final, Optional, Sequence, TypeVar

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from ...core import Shape
from .test import GenTestIn, TestError
from .utils import NnData, NnSample, download_root


def get_torch_dataset(ds, **kwargs) -> NnData:
    with open(os.devnull, "w") as devnull:
        sys.stdout, real_stdout = devnull, sys.stdout
        sys.stderr, real_stderr = devnull, sys.stderr

        result = ds(
            root=download_root(),
            download=True,
            **kwargs,
        )

        sys.stdout = real_stdout
        sys.stderr = real_stderr

    return [(torch.as_tensor(x), torch.as_tensor(y)) for x, y in result]


T = TypeVar("T")


def _shuffle(s: Sequence[T]) -> Sequence[T]:
    rng = torch.Generator().manual_seed(0xC0FFEE)
    perm = torch.randperm(len(s), generator=rng)
    return [s[i] for i in perm]


class NnDataMixin(ABC):
    @classmethod
    @abstractmethod
    def raw_data(cls, train: bool = True) -> NnData:
        raise NotImplementedError

    @classmethod
    def test_data(cls) -> NnData:
        return _shuffle(cls.raw_data(train=False))

    @classmethod
    def train_data(cls) -> NnData:
        return cls.raw_data(train=True)


_cifar_normalize: Final = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

_cifar_shape: Final = (3, 32, 32)


class Cifar10Mixin(NnDataMixin):
    input_shape: Final[Shape] = _cifar_shape
    num_classes: Optional[int] = 10

    data_transform: torch.nn.Module = transforms.Compose(
        [transforms.ToTensor(), _cifar_normalize]
    )

    @classmethod
    def raw_data(cls, train: bool = True) -> NnData:
        return get_torch_dataset(
            CIFAR10, train=train, transform=cls.data_transform
        )


class Cifar10TestIn(Cifar10Mixin, GenTestIn):
    pass


class Cifar10TestError(Cifar10Mixin, TestError):
    pass


class MnistMixin(NnDataMixin):
    input_shape: Final[Shape] = (28, 28)
    num_classes: Optional[int] = 10

    @classmethod
    def raw_data(cls, train: bool = True) -> NnData:
        return get_torch_dataset(
            MNIST, train=train, transform=transforms.ToTensor()
        )


class MnistTestIn(MnistMixin, GenTestIn):
    pass


class MnistTestError(MnistMixin, TestError):
    pass
