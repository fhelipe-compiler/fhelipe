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
from argparse import ArgumentParser
from pathlib import Path
from typing import (
    Dict,
    Final,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import tensorfhe as tfhe
import tensorfhe.app.actions as act
import torch
from tensorfhe import Input, Shape, TensorV, Vector


class ResNetWeights(act.PopulateShared):
    def __init__(self, layers: int = 20, **kwargs):
        self.__l = layers

    def __weight_tensors(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        tensors = {}
        for raw_name, tensor in state_dict.items():
            name = raw_name.rpartition("module.")[2]
            tensors[name] = tensor

        return tensors

    def shared_in(self, exe: tfhe.app.ExeManager) -> Mapping[str, TensorV]:
        path = Path(__file__).parent / "weights" / f"resnet{self.__l}.pt"
        raw_weights = torch.load(path, map_location=torch.device("cpu"))
        return self.__weight_tensors(raw_weights["state_dict"])


class ResNetError(act.ArgMaxOutputMixin, act.Cifar10TestError):
    pass


TensorT = TypeVar("TensorT", bound="TensorLike")


class TensorLike(Protocol):
    @property
    def shape(self) -> Shape:
        ...

    def __add__(self: TensorT, other: TensorT) -> TensorT:
        ...


class BaseResNet(tfhe.App, ABC, Generic[TensorT]):
    actions = {
        *act.base_actions,
        act.Cifar10TestIn,
        ResNetError,
        ResNetWeights,
    }

    @classmethod
    def add_instance_cli_args(cls, parser: ArgumentParser) -> None:
        g = parser.add_argument_group("ResNet Options")
        g.add_argument(
            "+l",
            "++layers",
            choices=(20, 32, 44, 56, 110),
            type=int,
        )
        g.add_argument(
            "++relu-scale",
            type=int,
        )

    def weights(self, name: str, shape: Shape) -> Input:
        return tfhe.public_in(name, shape)

    def bn_mul(self, name: str, k: int) -> Input:
        weight = self.weights(name + ".weight", (k,))
        var = self.weights(name + ".running_var", (k,))
        eps = 1e-05

        return weight / (var + eps).sqrt()

    def bn_add(self, name: str, k: int) -> Input:
        mul = self.bn_mul(name, k)
        bias = self.weights(name + ".bias", (k,))
        mean = self.weights(name + ".running_mean", (k,))

        return bias - mean * mul

    @abstractmethod
    def input_img(self, name: str, shape: Shape) -> TensorT:
        raise NotImplementedError

    @abstractmethod
    def relu(self, x: TensorT, name: str) -> TensorT:
        raise NotImplementedError

    @abstractmethod
    def conv_bn(
        self, img: TensorT, c_name: str, bn_name: str, k: int, stride: int = 1
    ) -> TensorT:
        raise NotImplementedError

    @abstractmethod
    def downsample(self, img: TensorT) -> TensorT:
        raise NotImplementedError

    @abstractmethod
    def average_pool(self, img: TensorT) -> TensorT:
        raise NotImplementedError

    @abstractmethod
    def fc(self, img: TensorT, out_c: int) -> TensorT:
        raise NotImplementedError

    @abstractmethod
    def to_value(self, x: TensorT) -> Vector:
        raise NotImplementedError

    def save_value(self, name: str, x: TensorT) -> TensorT:
        self.__out[name] = self.to_value(x)
        return x

    def __basic_block(self, img: TensorT, name: str, stride=1) -> TensorT:
        x = img
        k = x.shape[0] * stride

        x = self.conv_bn(x, name + ".conv1", name + ".bn1", k, stride=stride)
        x = self.relu(x, name + ".relu1")
        x = self.conv_bn(x, name + ".conv2", name + ".bn2", k)

        if stride == 1:
            sc = img
        elif stride == 2:
            sc = self.downsample(img)
        else:
            raise ValueError

        self.save_value(name + ".shortcut", sc)

        return self.relu(x + sc, name + ".relu2")

    def __first_conv(self, img: TensorT) -> TensorT:
        img = self.conv_bn(img, "conv1", "bn1", 16)
        img = self.relu(img, "relu1")
        return img

    def __init__(
        self,
        *,
        layers: int = 20,
        relu_scale: int = 40,
        N: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.relu_scale: Final = relu_scale

        id: Tuple[int, ...] = (layers, relu_scale)
        if N is not None:
            id += (N,)

        self.__out: Dict[str, Vector] = {}

        x = self.input_img(tfhe.stdin, act.Cifar10TestIn.input_shape)
        x = self.__first_conv(x)

        n = (layers - 2) // 6
        for i in range(3):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                x = self.__basic_block(x, f"layer{i + 1}.{j}", stride)

        x = self.average_pool(x)
        self.save_value("pool", x)

        x = self.fc(x, 10)
        self.save_value(tfhe.stdout, x)

        super().__init__(id=id, out=self.__out)
