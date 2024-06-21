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
from math import prod
from typing import Final, List, Sequence, Tuple

import numpy as np
import tensorfhe as tfhe
import tensorfhe.app.actions as act
from fheapps.resnet.base_resnet import BaseResNet
from tensorfhe import Input, Shape, Vector


def sum_rot(vec: Vector, cnt: int, stride: int) -> Vector:
    i = 1
    while i < cnt:
        rot = vec.rotate(i * stride)
        vec += rot
        i *= 2

    if i != cnt:
        raise ValueError(i, cnt)

    return vec


class Tensor:
    def __init__(self, vec: Vector, c: int, h: int, s: int):
        self.vec = vec
        # `c x h x h` image tensor
        # each channel is strided by `s`
        # There are `len(vec) / (c x h x h)` copies of the tensor
        self.c = c
        self.h = h
        self.s = s

        if c % (s * s) != 0:
            raise ValueError
        if len(vec) < c * h * h:
            raise ValueError

    def __len__(self) -> int:
        return prod(self.shape)

    @property
    def shape(self) -> Shape:
        return (self.c, self.h, self.h)

    @staticmethod
    def like(vec: Vector, t: "Tensor") -> "Tensor":
        return Tensor(vec, t.c, t.h, t.s)

    def sum_slots(self, cnt: int, stride: int) -> None:
        self.vec = sum_rot(self.vec, cnt, -stride)

    def replicate(self, N: int) -> None:
        self.vec = sum_rot(self.vec, N // len(self), len(self))

    def __add__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.c != other.c or self.h != other.h or self.s != other.s:
            raise ValueError

        return Tensor.like(self.vec + other.vec, self)


class ManualResNet(BaseResNet[Tensor]):
    @classmethod
    def add_instance_cli_args(cls, parser: ArgumentParser) -> None:
        super(ManualResNet, cls).add_instance_cli_args(parser)
        parser.add_argument("+N", type=int, help="Vector size")

    def __multiplex_ind(self, i: int, s: int) -> Tuple[int, ...]:
        return (i // (s * s), i // s % s, i % s)

    def __multiplex(self, in_: Input, s: int) -> Input:
        c = in_.shape[-1]
        cm = c // (s * s)

        mult_shape = in_.shape[:-1] + (cm, s, s)
        mult = tfhe.zeros(mult_shape)

        for i in range(c):
            mult = mult.put[(..., *self.__multiplex_ind(i, s))](in_[..., i])

        return mult

    def __channel_mask(self, i: int, h: int, s: int) -> Vector:
        hs = h * s
        mask = tfhe.zeros((self.N // (hs * hs), s, s))
        mask = mask.put[self.__multiplex_ind(i, s)](1)
        mask = mask.replicate(dim=1, n=h).replicate(dim=3, n=h)
        return self.__enc(mask)

    def __channel_offset(self, i: int, h: int, s: int) -> int:
        hs = h * s
        i_1, i_2, i_3 = self.__multiplex_ind(i, s)
        return i_1 * hs * hs + i_2 * hs + i_3

    def __enc(self, in_: Input) -> Vector:
        return in_.flatten().extend(dim=0, size=self.N).enc_vector()

    def input_img(self, name: str, shape: Shape) -> Tensor:
        raw_in = tfhe.secret_in(name, shape)
        raw_in = raw_in.extend(dim=0, size=4)

        copy_cnt = self.N // len(raw_in)
        replicated = raw_in.replicate(dim=0, n=copy_cnt)
        replicated /= self.relu_scale

        return Tensor(self.__enc(replicated), 4, shape[1], 1)

    def relu(self, x: Tensor, name: str) -> Tensor:
        p1, p2, p3 = tfhe.lib.sign_polynomials(alpha=13)

        s = x.vec
        s = p1(s)._bootstrap()
        s = p2(s)._bootstrap()
        s = p3(s)

        x.vec = tfhe.lib.relu_from_sign(x.vec, s)._bootstrap()

        return self.save_value(name, x)

    def to_value(self, x: Tensor) -> Vector:
        return x.vec

    def __conv_shift_products(
        self, img: Tensor, w: Input, i_h: int, i_w: int
    ) -> Sequence[Vector]:
        k, c, _, _ = w.shape
        h, s = img.h, img.s

        s_h = -i_h + 1
        s_w = -i_w + 1
        img_rot = s_h * h * s**2 + s_w * s
        img = Tensor.like(img.vec.rotate(img_rot), img)

        w = w[..., i_h, i_w]
        w = self.__multiplex(w, s)

        w = (
            w.replicate(dim=2, n=h)
            .replicate(dim=4, n=h)
            .put[tfhe.utils.shift_wrap_selection(2, s_h)](0)
            .put[tfhe.utils.shift_wrap_selection(4, s_w)](0)
        )

        k_per_v = self.N // (c * h * h)

        w_in = (w[i : i + k_per_v] for i in range(0, k, k_per_v))
        w_vec = (self.__enc(w_v) for w_v in w_in)

        return [img.vec * w_v for w_v in w_vec]

    def __conv_extract_channel(
        self,
        sum_t: Sequence[Tensor],
        j: int,
        mul: Input,
        c: int,
        h: int,
        s: int,
        stride: int,
    ) -> Vector:
        h_o, s_o = h // stride, s * stride

        global_offset = j * c * h * h
        t = sum_t[global_offset // self.N]
        t_offset = global_offset % self.N

        out_offset = self.__channel_offset(j, h_o, s_o)

        t_rot = t.vec.rotate(-t_offset + out_offset)

        mask = self.__channel_mask(j, h_o, s_o) * mul
        return t_rot * mask

    def __bn_add(self, name: str, k: int, h: int, s: int) -> Vector:
        in_ = self.bn_add(name, k)
        in_ /= self.relu_scale
        in_ = self.__multiplex(in_, s)
        in_ = in_.replicate(dim=1, n=h).replicate(dim=3, n=h)
        in_ = in_.replicate(dim=0, n=self.N // len(in_))
        return self.__enc(in_)

    def conv_bn(
        self, img: Tensor, conv_name: str, bn_name: str, k: int, stride: int = 1
    ) -> Tensor:
        c, h, s = img.c, img.h, img.s
        h_o, s_o = h // stride, s * stride
        if k % (stride * s) ** 2 != 0:
            raise ValueError

        if c == 4:
            w = self.weights(conv_name + ".weight", (k, 3, 3, 3))
            w = w.extend(dim=1, size=4)
        else:
            w = self.weights(conv_name + ".weight", (k, c, 3, 3))

        products = [
            self.__conv_shift_products(img, w, i_h, i_w)
            for i_h in range(3)
            for i_w in range(3)
        ]
        product_sums: List[Vector] = [sum(x) for x in zip(*products)]
        sum_t = [Tensor.like(s, img) for s in product_sums]

        for t in sum_t:
            t.sum_slots(s, 1)
            t.sum_slots(s, h * s)
            t.sum_slots(c // s**2, (h * s) ** 2)

        bn_mul = self.bn_mul(bn_name, k)
        out_ch = [
            self.__conv_extract_channel(sum_t, i, bn_mul[i], c, h, s, stride)
            for i in range(k)
        ]

        out_ch_sum = sum(out_ch)
        assert not isinstance(out_ch_sum, int)

        out_t = Tensor(out_ch_sum, k, h_o, s_o)
        out_t.replicate(self.N)
        out_t.vec += self.__bn_add(bn_name, k, h_o, s_o)

        return self.save_value(bn_name, out_t)

    def downsample(self, img: Tensor) -> Tensor:
        c_i, h_i, s_i = img.c, img.h, img.s
        c_o, h_o, s_o = c_i * 2, h_i // 2, s_i * 2

        out_channels = []
        for j_i in range(c_i):
            j_o = j_i + c_i // 2

            offset_i = self.__channel_offset(j_i, h_i, s_i)
            offset_o = self.__channel_offset(j_o, h_o, s_o)

            rot_v = img.vec.rotate(-offset_i + offset_o)
            mask = self.__channel_mask(j_o, h_o, s_o)
            out_channels.append(rot_v * mask)

        out_channels_sum = sum(out_channels)
        assert not isinstance(out_channels_sum, int)

        out_t = Tensor(out_channels_sum, c_o, h_o, s_o)
        out_t.replicate(self.N)

        return out_t

    def average_pool(self, img: Tensor) -> Tensor:
        c, h, s = img.c, img.h, img.s
        img.sum_slots(h, s)
        img.sum_slots(h, h * s * s)

        chan_slices = []
        for j in range(0, c, s):
            offset_i = self.__channel_offset(j, h, s)
            offset_o = j

            rot_v = img.vec.rotate(-offset_i + offset_o)

            mask = tfhe.zeros((self.N,)).put[j : j + s](1 / (h * h))
            chan_slices.append(rot_v * mask)

        slice_sum = sum(chan_slices)
        assert not isinstance(slice_sum, int)
        return Tensor(slice_sum, c, 1, 1)

    def __fc_diagonals(self, c_o: int, c_i: int) -> Sequence[Vector]:
        m = self.weights("linear.weight", (c_o, c_i))

        diagonals = []
        d_cnt = c_o + c_i - 1
        for i in range(c_o + c_i - 1):
            d = tfhe.zeros((c_i,))
            for j in range(c_i):
                row = (c_o - 1) - i + j
                if 0 <= row < c_o:
                    d = d.put[j](m[row, j])

            diagonals.append(self.__enc(d))

        return diagonals

    def fc(self, img: Tensor, c_o: int) -> Tensor:
        c_i = img.c

        diagonals = self.__fc_diagonals(c_o, c_i)
        prod = [d * img.vec for d in diagonals]
        res_v = sum(p.rotate((c_o - 1) - i) for i, p in enumerate(prod))

        bias = self.weights("linear.bias", (c_o,)) / self.relu_scale
        res_v += self.__enc(bias)

        return Tensor(res_v, c_o, 1, 1)

    def __init__(self, N: int = 2**15, **kwargs) -> None:
        self.N: Final = N
        super().__init__(N=N, **kwargs)


if __name__ == "__main__":
    ManualResNet.main()
