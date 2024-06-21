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

from typing import Tuple, overload

from ..core import Input, InputT, Tensor, TensorT, _result_repack
from .utils import conv_shifts, dim_selection


def conv_zero_weights(
    w: InputT, dim: int, kernel: int, pad: int, shift: int
) -> InputT:
    size_i = w.shape[dim]
    size_o = size_i + 2 * pad - (kernel - 1)

    start = max(0, shift)
    end = min(size_o, size_i + shift)

    w = w.put[dim_selection(dim, slice(None, start))](0)
    w = w.put[dim_selection(dim, slice(end, None))](0)

    return w


@overload
def conv2d_raw(
    image: InputT,
    weights: InputT,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> InputT:
    ...


@overload
def conv2d_raw(
    image: Tensor,
    weights: Input,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> Tensor:
    ...


@_result_repack
def conv2d_raw(
    image: Tensor,
    weights: Input,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> Tensor:
    """
    Convolve image with weights

    Args:
        image: shape is C x H x W
        weights: shape is K x C x R x S
        stride: must be powers of 2
        padding: must be <= ((R-1) // 2, (S-1) // 2)
    Returns:
        Input if image is an Input; Tensor otherwise.
        If padding produces a smaller image, the output is zero-padded.
    """
    _, h, w = image.shape
    k, c, r, s = weights.shape
    p_r, p_s = padding

    image_ext, weights_ext = [], []
    for i in range(r):
        s_i = p_r - i
        image_i = image.shift(dim=-2, by=s_i)

        for j in range(s):
            s_j = p_s - j
            image_ij = image_i.shift(dim=-1, by=s_j)
            image_ext.append(image_ij.replicate(dim=0, n=k))

            w_ij = weights[:, :, i, j]
            w_ij = w_ij.reshape((k, c, 1, 1)).broadcast_to((k, c, h, w))
            w_ij = conv_zero_weights(w_ij, 2, r, p_r, s_i)
            w_ij = conv_zero_weights(w_ij, 3, s, p_s, s_j)
            weights_ext.append(w_ij)

    products = [img * wgt for img, wgt in zip(image_ext, weights_ext)]
    products_sum = sum(products)

    assert not isinstance(products_sum, int)  # for mypy

    for i, s in enumerate(stride):
        products_sum = products_sum.stride(dim=i + 2, by=s)

    result = products_sum.sum(dim=1)
    return result


@overload
def conv2d(image: InputT, weights: InputT, stride: int = 1) -> InputT:
    ...


@overload
def conv2d(image: Tensor, weights: Input, stride: int = 1) -> Tensor:
    ...


def conv2d(image: Tensor, weights: Input, stride: int = 1):
    _, _, r, s = weights.shape
    return conv2d_raw(
        image,
        weights,
        stride=(stride, stride),
        padding=((r - 1) // 2, (s - 1) // 2),
    )


@overload
def conv2d_unpadded(image: InputT, weights: InputT, stride: int = 1) -> InputT:
    ...


@overload
def conv2d_unpadded(image: Tensor, weights: Input, stride: int = 1) -> Tensor:
    ...


def conv2d_unpadded(image: Tensor, weights: Input, stride: int = 1):
    return conv2d_raw(image, weights, stride=(stride, stride), padding=(0, 0))
