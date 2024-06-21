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
from fheapps.tensor import mul_tm


class Relu(tfhe.App):
    @classmethod
    def add_instance_cli_args(cls, parser):
        parser.add_argument("+N", type=int, help="Ciphertext size")
        parser.add_argument(
            "+a",
            "++alpha",
            type=int,
            choices=[8, 12, 13, 14],
            help="Precision parameter",
        )

    def __init__(
        self,
        *,
        alpha: int = 12,
        N: int = 2**15,
        **kwargs,
    ):
        x_in = tfhe.secret_in(tfhe.stdin, (N,))
        x_in = x_in * 2 - 1
        x = x_in.enc_vector()

        s = x
        for p in tfhe.lib.sign_polynomials(alpha):
            s = p(s)._bootstrap()

        y = tfhe.lib.relu_from_sign(x, s)

        super().__init__(id=(alpha, N), out=y)


if __name__ == "__main__":
    Relu.main()
