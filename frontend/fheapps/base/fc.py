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


class Fc(tfhe.App):
    @classmethod
    def add_instance_cli_args(cls, parser):
        g = parser.add_argument_group("Fully Connected Options")
        g.add_argument("+n", type=int)
        g.add_argument("+m", type=int)

    def __init__(self, n: int = 32, m: int = 64, **kwargs):
        v = tfhe.tensor("vector", (m,))
        wgt = tfhe.public_in("weights", (n, m))
        result = tfhe.lib.mul_mv(wgt, v)

        super().__init__(id=(n, m), out=result)


if __name__ == "__main__":
    Fc.main()
