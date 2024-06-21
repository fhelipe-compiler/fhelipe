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

from typing import List, Tuple

import tensorfhe as tfhe
import torch
from tensorfhe import TensorT
from tensorfhe.lib import conv_shifts, shift_wrap_selection


def max_pool2d(
    image: TensorT, kernel_shape: Tuple[int, int], stride: int
) -> TensorT:
    r, s = kernel_shape
    image_ext: List[TensorT] = []
    for s_i in conv_shifts(r):
        image_i = image.shift(dim=1, by=s_i)
        for s_j in conv_shifts(s):
            image_ij = image_i.shift(dim=2, by=s_j)

            zero_out = (
                tfhe.as_input(torch.ones(image.shape))
                .put[shift_wrap_selection(1, s_i)](0)
                .put[shift_wrap_selection(2, s_j)](0)
            )

            # nsamar: -1 is the smallest negative number, because maxpool2d
            #         assumes its inputs are in (-1, 1)
            min_along_edges = (
                tfhe.as_input(torch.zeros(image.shape))
                .put[shift_wrap_selection(1, s_i)](-1)
                .put[shift_wrap_selection(2, s_j)](-1)
            )

            image_ext.append(image_ij * zero_out + min_along_edges)

    raw_result = tfhe.lib.maximum(image_ext)
    result = raw_result.stride(dim=1, by=stride).stride(dim=2, by=stride)
    return result


class MaxPool(tfhe.App):
    @classmethod
    def add_instance_cli_args(cls, parser):
        g = parser.add_argument_group("Pooling Options")
        g.add_argument("+c", type=int)
        g.add_argument("+w", type=int)
        g.add_argument("+r", "++filtter-size", dest="r", type=int)
        g.add_argument("+s", "++stride", type=int)

    def __init__(
        self, c: int = 4, w: int = 32, r: int = 3, stride: int = 1, **kwargs
    ):
        img = tfhe.tensor("image", (c, w, w))
        result = max_pool2d(img, (r, r), stride)

        super().__init__(id=(c, w, r, stride), out=result)


if __name__ == "__main__":
    MaxPool.main()
