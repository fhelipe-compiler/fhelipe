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

from typing import Tuple, Union

import tensorfhe as tfhe


@tfhe._manual_repack
def mttkrp(b, c, d):
    i, l, k = b.shape
    j, _ = c.shape

    b_rep = b.replicate(dim=2, n=j)
    c_rep = c.replicate(dim=0, n=l).replicate(dim=0, n=i)
    bc = (b_rep * c_rep).sum(dim=3)._chet_repack()  # i, l, j

    d_rep = d.replicate(dim=0, n=i)
    bcd = (bc * d_rep).sum(dim=1)._chet_repack()
    return bcd


class MTTKRP(tfhe.App):
    @classmethod
    def add_instance_cli_args(cls, parser):
        dim_group = parser.add_argument_group(
            title="Tensor Dimensions",
            description="Multiplying an [i x j x l] tensor by an [l x k] matrix",
        )
        dim_group.add_argument("+i", type=int)
        dim_group.add_argument("+j", type=int)
        dim_group.add_argument("+k", type=int)
        dim_group.add_argument("+l", type=int)

        pub_group = parser.add_argument_group("Ct/Pt Inputs")
        pub_group.add_argument("++pub-b", action="store_false", dest="secret_b")
        pub_group.add_argument("++pub-c", action="store_false", dest="secret_c")
        pub_group.add_argument("++pub-d", action="store_false", dest="secret_d")

    def __init__(
        self,
        *,
        i: int = 64,
        j: int = 64,
        k: int = 64,
        l: int = 64,
        secret_b: bool = True,
        secret_c: bool = True,
        secret_d: bool = True,
        **kwargs,
    ):
        if not (secret_b or secret_c or secret_d):
            raise ValueError

        id: Tuple[Union[str, int], ...] = (i, j, k, l)
        if not secret_b:
            id += ("pb",)
        if not secret_c:
            id += ("pc",)
        if not secret_d:
            id += ("pd",)

        b = tfhe.input("b", (i, l, k), secret=secret_b).enc()
        c = tfhe.input("c", (j, k), secret=secret_c).enc()
        d = tfhe.input("d", (l, j), secret=secret_d).enc()

        super().__init__(id=id, out=mttkrp(b, c, d))


if __name__ == "__main__":
    MTTKRP.main()
