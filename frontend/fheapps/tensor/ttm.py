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


@tfhe._result_repack
def mul_tm(t, m):
    i, j, l = t.shape
    k, _ = m.shape

    t_rep = t.replicate(dim=2, n=k)
    m_rep = m.replicate(dim=0, n=j).replicate(dim=0, n=i)
    prod = t_rep * m_rep
    return prod.sum(dim=3)


class TTM(tfhe.App):
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

        pub_group = parser.add_argument_group("Ct/Pt inputs")
        pub_group.add_argument("++pub-t", action="store_false", dest="secret_t")
        pub_group.add_argument("++pub-m", action="store_false", dest="secret_m")

    def __init__(
        self,
        *,
        i: int = 64,
        j: int = 64,
        k: int = 64,
        l: int = 64,
        secret_t: bool = True,
        secret_m: bool = True,
        **kwargs,
    ):
        if not secret_t and not secret_m:
            raise ValueError

        id: Tuple[Union[str, int], ...] = (i, j, k, l)
        if not secret_t:
            id += ("pt",)
        if not secret_m:
            id += ("pm",)

        t = tfhe.input("t", (i, j, k), secret=secret_t).enc()
        m = tfhe.input("m", (l, k), secret=secret_m).enc()

        super().__init__(id=id, out=mul_tm(t, m))


if __name__ == "__main__":
    TTM.main()
