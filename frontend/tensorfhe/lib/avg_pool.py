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

from ..core import TensorT, _result_repack, as_input
from .utils import conv_shifts, shift_wrap_selection


def __sum_pool_1d(image: TensorT, kernel_size: int, dim: int = -1):
    shifted = [image.shift(dim=dim, by=-i) for i in range(kernel_size)]
    result = sum(shifted)
    assert not isinstance(result, int)  # for mypy

    return result


@_result_repack
def avg_pool2d(image: TensorT, kernel_size: int, stride: int) -> TensorT:
    products = []

    w = as_input(1 / kernel_size**2).broadcast_to(image.shape)
    dims = len(image.shape)
    for s_i in conv_shifts(kernel_size):
        image_i = image.shift(dim=-2, by=s_i)
        w_i = w.put[shift_wrap_selection(dims - 2, s_i)](0)

        for s_j in conv_shifts(kernel_size):
            image_ij = image_i.shift(dim=-1, by=s_j)
            w_ij = w_i.put[shift_wrap_selection(dims - 1, s_j)](0)

            products.append(image_ij * w_ij)

    s = sum(products)
    assert not isinstance(s, int)

    return s.stride(dim=-1, by=stride).stride(dim=-2, by=stride)


@_result_repack
def global_avg_pool2d(img: TensorT) -> TensorT:
    scale = img.shape[-1] * img.shape[-2]
    result = img.sum(dim=-1).sum(dim=-1) * (1 / scale)
    return result
