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


@tfhe._result_repack
def inner_prod(a, b):
    prod = a * b
    for _ in a.shape:
        prod = prod.sum(dim=0)
    return prod


class InnerProd(tfhe.App):
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
        parser.add_argument("++pub", action="store_false", dest="secret_b")

    def __init__(self, shape: tfhe.Shape, secret_b: bool = True, **kwargs):
        a = tfhe.tensor("a", shape)
        b = tfhe.input("b", shape, secret=secret_b).enc()

        super().__init__(id=shape, out=inner_prod(a, b))


if __name__ == "__main__":
    InnerProd.main()
