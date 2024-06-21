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
def mul_tv(t, v):
    i, j, k = t.shape
    v_rep = v.replicate(dim=0, n=j).replicate(dim=0, n=i)
    prod = t * v_rep
    return prod.sum(dim=2)


class MulTV(tfhe.App):
    @classmethod
    def add_instance_cli_args(cls, parser):
        dim_group = parser.add_argument_group(
            title="Tensor Dimensions",
            description="Multiplying an [i x j x k] tensor by a [k] vector",
        )
        dim_group.add_argument("+i", type=int)
        dim_group.add_argument("+j", type=int)
        dim_group.add_argument("+k", type=int)

        pub_group = parser.add_argument_group("Ct/Pt inputs")
        pub_group.add_argument("++pub-t", action="store_false", dest="secret_t")
        pub_group.add_argument("++pub-v", action="store_false", dest="secret_v")

    def __init__(
        self,
        *,
        i: int = 2,
        j: int = 2,
        k: int = 2,
        secret_t: bool = True,
        secret_v: bool = True,
        **kwargs,
    ):
        if not secret_t and not secret_v:
            raise ValueError

        id = f"{i}_{j}_{k}"
        if not secret_t:
            id = "pt_" + self.id
        if not secret_v:
            id = "pv_" + self.id

        t = tfhe.input("t", (i, j, k), secret=secret_t).enc()
        v = tfhe.input("v", (k,), secret=secret_v).enc()

        super().__init__(out=mul_tv(t, v), id=id)


if __name__ == "__main__":
    MulTV.main()
