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
from fheapps.lola.base_mnist import BaseLolaMnist
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


def fc_stride_w(w: Input, stride: int, groups: int) -> Input:
    n, m = w.shape

    blocks = [w[:, i::groups] for i in range(groups)]
    blocks = [b.extend(dim=-1, size=stride * groups) for b in blocks]
    blocks = [b.rotate(dim=-1, by=i * stride) for i, b in enumerate(blocks)]

    w_new = sum(blocks)
    assert not isinstance(w_new, int)
    return w_new


class ManualLolaMnist(BaseLolaMnist[Vector]):
    N = 2**14
    fc1_copy = 16

    def enc(self, x: Input) -> Vector:
        # 2 * N because the backend expects 32K
        return enc_v(x, 2 * self.N).enc_vector()

    def conv(self, x: Vector, w: Input, b: Input) -> Vector:
        out_c = conv(x, w, b, 28, pad=1)
        return pack(out_c, 14 * 14)

    def fc1(self, x: Vector, w: Input, b: Input) -> Vector:
        w = w.extend(dim=2, size=14).extend(dim=3, size=14)
        w = fc_to_conv_order(w)
        w = w.reshape((100, 5 * 14 * 14))
        w = fc_pad_w(w, self.fc1_copy)

        n, m = w.shape
        l = len(x)

        prod = fc_products(x, w, l, self.fc1_copy)
        y = fc_sum(prod, m, l)

        b = fc_pad_bias(b, n, m, l, self.fc1_copy)
        y = y + b
        return y

    def fc2(self, x: Vector, w: Input, b: Input) -> Vector:
        groups = self.fc1_copy
        stride = self.N // self.fc1_copy

        w = fc_pad_w(w)
        w = fc_stride_w(w, stride, groups)

        n, m = w.shape
        l = len(x)

        prod = fc_products(x, w, l)
        y = fc_sum(prod, m, l, stride, groups)

        b = enc_v(b, l)
        y = y + b
        return y


if __name__ == "__main__":
    ManualLolaMnist.main()
