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

from typing import Sequence, Tuple

import tensorfhe as tfhe
import torch
from fheapps.lola.base_cifar import BaseLolaCifar
from fheapps.lola.manual_shared import (
    conv,
    enc_v,
    fc_pad_bias,
    fc_pad_w,
    fc_products,
    fc_sum,
    fc_to_conv_order,
    pack,
)
from tensorfhe import Input, Shape, Vector
from tensorfhe.core import op

T = Tuple[Vector, ...]


class ConvToFcOp(op.InputOp):
    def __init__(self, w: op.Op, *, h_in: int, stride: int, pad: int):
        self.h_in = h_in
        self.stride = stride
        self.pad = pad

        k, c, r, s = w.shape
        if r != s:
            raise ValueError

        h_in_effective = h_in - (r - 1) + 2 * pad
        self.h_out = (h_in_effective + stride - 1) // stride

        out_shape = (k, self.h_out, self.h_out, c, h_in, h_in)
        super().__init__(w, shape=out_shape)

    def evaluate(self, *parents: tfhe.TensorV) -> tfhe.TensorV:
        t = parents[0]
        k, c, r, s = t.size()

        h_in, h_out = self.h_in, self.h_out
        stride, pad = self.stride, self.pad

        w_new = torch.zeros((k, h_out, h_out, c, h_in, h_in), dtype=t.dtype)
        for i in range(h_out):
            for j in range(h_out):
                for si in range(r):
                    for sj in range(s):
                        i1 = stride * i + si - pad
                        j1 = stride * j + sj - pad

                        if i1 < 0 or j1 < 0 or i1 >= h_in or j1 >= h_in:
                            continue

                        w_new[:, i, j, :, i1, j1] = t[:, :, si, sj]

        return w_new


def conv_to_fc(w: Input, h_in: int, stride: int = 2, pad: int = 0) -> Input:
    return w.apply(ConvToFcOp, h_in=h_in, stride=stride, pad=pad)


class ManualLolaCifar(BaseLolaCifar[T]):
    N = 2**14

    @property
    def zeros(self) -> Vector:
        return tfhe.zeros((2 * self.N,))

    def enc(self, x: Input) -> T:
        # 2 * N because the backend expects 32K
        return (enc_v(x, 2 * self.N).enc_vector(),)

    def conv1(self, x: T, w: Input, b: Input) -> T:
        out_c = conv(x[0], w, b, 32, pad=1)

        k_mid = len(out_c) // 2
        pack_s = 16 * 16

        return (pack(out_c[:k_mid], pack_s), pack(out_c[k_mid:], pack_s))
        # return (pack(out_c, pack_s),)

    def conv2(self, x: T, w: Input, b: Input) -> T:
        h_in_raw = 16
        h_in = 14
        k, c, _, _ = w.shape

        w = conv_to_fc(w, h_in=h_in, stride=2, pad=4)
        # w = w.extend(dim=1, size=7).extend(dim=2, size=7)
        h_out = w.shape[1]

        w = w.extend(dim=-1, size=h_in_raw).extend(dim=-2, size=h_in_raw)
        w = w.reshape((k * h_out * h_out, c, h_in_raw, h_in_raw))
        self.save("conv-w", (w,))

        w = fc_to_conv_order(w)
        w = w.reshape((k * h_out * h_out, c * h_in_raw * h_in_raw))

        n, m = w.shape
        m_mid = c // 2 * h_in_raw * h_in_raw
        l = len(x[0])

        ws = (w[:, :m_mid], w[:, m_mid:])
        prods = tuple(fc_products(xi, wi, l) for xi, wi in zip(x, ws))
        prods_sum = [p1 + p2 for p1, p2 in zip(*prods)]
        y = fc_sum(prods_sum, self.N, l)

        # prods_sum = fc_products(x[0], w, l)
        # y = fc_sum(prods_sum, l, l)

        b = b.replicate(dim=1, n=h_out * h_out).reshape((k * h_out * h_out,))
        b = enc_v(b, l)
        y = y + b
        return (y,)

    def fc(self, x: T, w: Input, b: Input) -> T:
        w = w.reshape((10, 112 * 7 * 7))

        n, m = w.shape
        l = len(x[0])

        prod = fc_products(x[0], w, l)
        y = fc_sum(prod, m, l)
        b = enc_v(b, l)
        y = y + b
        return (y,)

    def square(self, x: T) -> T:
        return tuple(xi * xi for xi in x)

    def dec(self, x: T) -> Vector:
        v = sum(xi.rotate(i * self.N) for i, xi in enumerate(x))
        assert not isinstance(v, int)
        return v


if __name__ == "__main__":
    ManualLolaCifar.main()
