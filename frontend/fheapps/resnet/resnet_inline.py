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

from argparse import ArgumentParser
from typing import Final

import tensorfhe as fhelipe
from tensorfhe import Input, Shape, Tensor


class ResNet(fhelipe.App):
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
        return fhelipe.public_in(name, shape)

    def relu(self, x: Tensor, name: str) -> Tensor:
        return fhelipe.lib.relu(x, alpha=13)

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

    def input_img(self, name: str, shape: Shape) -> Tensor:
        in_ = fhelipe.secret_in(name, shape) / self.relu_scale
        return in_.enc()

    def __bn(self, img: Tensor, name: str) -> Tensor:
        k = img.shape[-3]
        pt_shape = (k, 1, 1)
        mul = self.bn_mul(name, k).reshape(pt_shape)
        add = self.bn_add(name, k).reshape(pt_shape) / self.relu_scale

        return img * mul + add

    def __conv(self, img: Tensor, name: str, k: int, stride: int = 1) -> Tensor:
        wgt = self.weights(name + ".weight", (k, img.shape[0], 3, 3))
        return fhelipe.lib.conv2d(img, wgt, stride)

    def conv_bn(
        self, img: Tensor, conv_name: str, bn_name: str, k: int, stride: int = 1
    ) -> Tensor:
        img = self.__conv(img, conv_name, k, stride)
        img = self.__bn(img, bn_name)
        return img

    def downsample(self, img: Tensor) -> Tensor:
        k = img.shape[0] * 2

        img = img.stride(dim=1, by=2).stride(dim=2, by=2)
        img = img.extend(dim=0, size=k).rotate(dim=0, by=k // 4)

        return img

    def average_pool(self, img: Tensor) -> Tensor:
        return fhelipe.lib.global_avg_pool2d(img)

    def fc(self, x: Tensor, out_c: int) -> Tensor:
        wgt = self.weights("linear.weight", (out_c, x.shape[0]))
        bias = self.weights("linear.bias", (out_c,))

        return fhelipe.lib.mul_mv(wgt, x) + bias / self.relu_scale

    def __basic_block(self, img: Tensor, name: str, stride=1) -> Tensor:
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

        return self.relu(x + sc, name + ".relu2")

    def __first_conv(self, img: Tensor) -> Tensor:
        img = self.conv_bn(img, "conv1", "bn1", 16)
        img = self.relu(img, "relu1")
        return img

    def __init__(
        self,
        *,
        layers: int = 20,
        relu_scale: int = 40,
        **kwargs,
    ) -> None:
        self.relu_scale: Final = relu_scale

        x = self.input_img(fhelipe.stdin, (3, 32, 32))
        x = self.__first_conv(x)

        n = (layers - 2) // 6
        for i in range(3):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                x = self.__basic_block(x, f"layer{i + 1}.{j}", stride)

        x = self.average_pool(x)
        x = self.fc(x, 10)

        super().__init__(id=(layers, relu_scale), out=x)


if __name__ == "__main__":
    ResNet.main()
