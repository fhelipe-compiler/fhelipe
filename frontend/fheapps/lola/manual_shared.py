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
from tensorfhe import Input, Vector
from tensorfhe.lib.conv import conv_zero_weights


def enc_v(w: Input, l: int) -> Input:
    return w.flatten().extend(dim=0, size=l)


def conv_interleave_block(h: int, k: int) -> Tuple[Input, Tuple[int, int]]:
    hs = h // 2
    block_h_sm = hs // 4
    block_start = 2 * (block_h_sm * k + min(k, hs % 4))
    block_size = 2 * (block_h_sm + (1 if k < hs % 4 else 0))

    block_mask = tfhe.zeros((h, h)).put[block_start : block_start + block_size](
        1
    )

    b_i = k // 2
    b_j = k % 2
    block_s = (-block_start + b_i, b_j)

    return block_mask, block_s


def preconv_1(x: Vector, h: int, r: int, pad: int, c: int, i: int, j: int):
    s_i = -(i - pad)
    s_j = -(j - pad)

    conv_mask = tfhe.ones((h, h))
    conv_mask = conv_zero_weights(conv_mask, 0, r, pad, s_i)
    conv_mask = conv_zero_weights(conv_mask, 1, r, pad, s_j)
    stride_mask = tfhe.ones((h, h)).put[1::2, :](0).put[:, 1::2](0)

    conv_r = -c * h * h + s_i * h + s_j

    blocks = []
    for k in range(4):
        block_mask, (s_i, s_j) = conv_interleave_block(h, k)
        block_r = s_i * h + s_j

        mask = conv_mask * stride_mask * block_mask
        flat_mask = enc_v(mask, len(x))

        # Masks are relative to position after conv rotation
        flat_mask = flat_mask.rotate(-conv_r)

        blocks.append((x * flat_mask).rotate(conv_r + block_r))

    return sum(blocks)


def conv(
    x: Vector, w: Input, b: Input, h: int, pad: int = 0
) -> Sequence[Vector]:
    k, c, r, s = w.shape
    l = len(x)
    assert r == s
    assert h % 4 == 0

    crs = c * r * s
    out_size = h * h // 4

    x_split = [
        preconv_1(x, h=h, r=r, pad=pad, c=ci, i=ri, j=si)
        for ci in range(c)
        for ri in range(r)
        for si in range(s)
    ]

    w = w.reshape((k, crs))
    w_ones = tfhe.ones((l,)).enc_vector()
    w_v = [[w_ones * w[ki, ci] for ci in range(crs)] for ki in range(k)]

    b_ones = tfhe.zeros((l,)).put[:out_size](1)
    b_v = [b_ones * b[ki] for ki in range(k)]

    out_c = [
        sum(w_v[ki][ci] * x_split[ci] for ci in range(crs)) + b_v[ki]
        for ki in range(k)
    ]
    return out_c


def pack(vs: Sequence[Vector], stride: int) -> Vector:
    packed = sum(v.rotate(i * stride) for i, v in enumerate(vs))
    assert not isinstance(packed, int)

    return packed


def fc_to_conv_order(w: Input) -> Input:
    n, c, hs, _ = w.shape
    h = 2 * hs

    assert h % 4 == 0

    w = w.insert_dim(dim=-1, size=2).insert_dim(dim=-3, size=2)
    w = w.reshape((n, c, h, h))

    w_new = tfhe.zeros(w.shape)
    for k in range(4):
        mask, (s_i, s_j) = conv_interleave_block(h, k)
        block = (w * mask).rotate(dim=-2, by=s_i).rotate(dim=-1, by=s_j)
        w_new += block

    w_new = w_new.shrink(dim=2, size=h // 4)
    return w_new


def replicate(x: Vector, stride: int, copy: int) -> Vector:
    i = 1
    while i < copy:
        x += x.rotate(stride * i)
        i *= 2
    return x


def sum_slots(x: Vector, stride: int, width: int) -> Vector:
    i = 1
    while i < width:
        x += x.rotate(-i * stride)
        i *= 2
    return x


def next_pow_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def ceil_n(n: int, copy: int) -> int:
    return n + (-n % copy)


def fc_pad_w(w: Input, copy: int = 1) -> Input:
    n, m = w.shape

    n = ceil_n(n, copy)
    m = next_pow_2(m)

    w = w.extend(dim=0, size=n)
    w = w.extend(dim=1, size=m)

    return w


def fc_products(x: Vector, w: Input, l: int, copy: int = 1) -> Sequence[Vector]:
    n, m = w.shape

    x = replicate(x, m, copy)

    w_v = [enc_v(w[i : i + copy], l) for i in range(0, n, copy)]

    return [x * vi for vi in w_v]
    # return [vi for vi in w_v]


def fc_sum(
    ys: Sequence[Vector], m: int, l: int, stride: int = 1, groups: int = 1
) -> Vector:
    ys = [sum_slots(y, 1, m // groups) for y in ys]
    ys = [sum_slots(y, stride, groups) for y in ys]
    # ys = [sum_slots(p, 1, m) for p in products]

    mask = tfhe.zeros((l,)).put[::m](1)
    ys = [y * mask for y in ys]
    ys = [y.rotate(i) for i, y in enumerate(ys)]

    y = sum(ys)
    assert not isinstance(y, int)

    return y


def fc_pad_bias(b: Input, n: int, m: int, l: int, copy: int = 1) -> Input:
    b = b.extend(dim=0, size=n)
    b = b.reshape((n // copy, copy)).T
    b = b.extend(dim=1, size=m)

    return enc_v(b, l)


def fc(x: Vector, w: Input, b: Input, copy: int = 1) -> Vector:
    w = fc_pad_w(w, copy)
    n, m = w.shape
    l = len(x)
    b = fc_pad_bias(b, n, m, l, copy)

    prod = fc_products(x, w, l, copy)
    y = fc_sum(prod, m, l)
    y = y + b
    return y
