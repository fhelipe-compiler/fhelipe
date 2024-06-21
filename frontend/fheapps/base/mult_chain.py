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

from typing import Sequence

import tensorfhe as tfhe
from tensorfhe import Shape, Tensor, VectorT


def compute_product_tree(values: Sequence[VectorT]) -> VectorT:
    if len(values) == 0:
        raise ValueError
    elif len(values) == 1:
        return values[0]
    elif len(values) == 2:
        return values[0] * values[1]
    else:
        midpoint = len(values) // 2
        return compute_product_tree(values[:midpoint]) * compute_product_tree(
            values[midpoint:]
        )


def multiplication_chain(
    shape: Shape, total_depth: int, total_width: int, total_chunk_depth: int
) -> Tensor:
    helper = tfhe.public_in("helper", shape)
    x = tfhe.tensor("x", shape)

    for depth in range(total_depth):
        sum = helper * x
        tmps = []
        for width in range(total_width - 1):
            tmp = helper * x
            for chunk_depth in range(total_chunk_depth - 1):
                tmp = helper * tmp
            tmps.append(tmp)
        x = compute_product_tree(tmps)
    return x


class MultiplicationChain(tfhe.App):
    @classmethod
    def add_instance_cli_args(cls, parser):
        parser.add_argument(
            "+d",
            dest="shape",
            type=int,
            action="append",
            metavar="DIM",
            required=True,
            help="""
                Size of input dimension. Repeat flag for multi-deminsional
                inputs.""",
        )
        parser.add_argument("++depth", default=10, type=int, dest="total_depth")
        parser.add_argument("++width", default=10, type=int, dest="total_width")
        parser.add_argument(
            "++chunk_depth", default=1, type=int, dest="total_chunk_depth"
        )

    def __init__(
        self,
        shape: Shape,
        total_depth: int,
        total_width: int,
        total_chunk_depth: int,
        **kwargs
    ):
        super().__init__(
            id=shape,
            out=multiplication_chain(
                shape, total_depth, total_width, total_chunk_depth
            ),
        )


if __name__ == "__main__":
    MultiplicationChain.main()
