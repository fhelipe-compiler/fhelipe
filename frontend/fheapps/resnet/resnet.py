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

import tensorfhe as tfhe
import tensorfhe.app.actions as act
from fheapps.resnet.base_resnet import BaseResNet
from tensorfhe import Shape, Tensor, _result_repack


class ResNet(BaseResNet[Tensor]):
    def input_img(self, name: str, shape: Shape) -> Tensor:
        in_ = tfhe.secret_in(name, shape) / self.relu_scale
        return in_.enc()

    def __bn(self, img: Tensor, name: str) -> Tensor:
        k = img.shape[-3]
        pt_shape = (k, 1, 1)
        mul = self.bn_mul(name, k).reshape(pt_shape)
        add = self.bn_add(name, k).reshape(pt_shape) / self.relu_scale

        result = img * mul + add

        return self.save_value(name, result)

    def __conv(self, img: Tensor, name: str, k: int, stride: int = 1) -> Tensor:
        wgt = self.weights(name + ".weight", (k, img.shape[0], 3, 3))
        result = tfhe.lib.conv2d(img, wgt, stride)

        return self.save_value(name, result)

    def conv_bn(
        self, img: Tensor, conv_name: str, bn_name: str, k: int, stride: int = 1
    ) -> Tensor:
        img = self.__conv(img, conv_name, k, stride)
        img = self.__bn(img, bn_name)
        return img

    @_result_repack
    def relu(self, x: Tensor, name: str) -> Tensor:
        p1, p2, p3 = tfhe.lib.sign_polynomials(alpha=13)

        s = x
        s = p1(s)._bootstrap()
        s = p2(s)._bootstrap()
        s = p3(s)

        x = tfhe.lib.relu_from_sign(x, s)._bootstrap()
        return self.save_value(name, x)

    @_result_repack
    def downsample(self, img: Tensor) -> Tensor:
        k = img.shape[0] * 2

        img = img.stride(dim=1, by=2).stride(dim=2, by=2)
        img = img.extend(dim=0, size=k).rotate(dim=0, by=k // 4)

        return img

    def average_pool(self, img: Tensor) -> Tensor:
        return tfhe.lib.global_avg_pool2d(img)

    def fc(self, x: Tensor, out_c: int) -> Tensor:
        wgt = self.weights("linear.weight", (out_c, x.shape[0]))
        bias = self.weights("linear.bias", (out_c,))

        return tfhe.lib.mul_mv(wgt, x) + bias / self.relu_scale

    def to_value(self, x: Tensor) -> Tensor:
        return x


if __name__ == "__main__":
    ResNet.main()
