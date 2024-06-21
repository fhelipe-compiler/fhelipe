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


class Conv(tfhe.App):
    @classmethod
    def add_instance_cli_args(cls, parser):
        g = parser.add_argument_group("Convolution Options")
        g.add_argument("+c", type=int, help="Number of input channels")
        g.add_argument("+k", "++output-channels", dest="k", type=int)
        g.add_argument("+w", type=int, help="Width and height of input image")
        g.add_argument(
            "+r",
            "++filtter-size",
            help="Width and height of filter",
            dest="r",
            type=int,
        )
        g.add_argument(
            "+s",
            "++stride",
            type=int,
            help="Stride; must be a power of 2",
        )

    def __init__(
        self,
        c: int = 3,
        k: int = 16,
        w: int = 32,
        r: int = 3,
        stride: int = 1,
        **kwargs,
    ):
        img = tfhe.tensor("image", (c, w, w))
        wgt = tfhe.public_in("weights", (k, c, r, r))
        result = tfhe.lib.conv2d(img, wgt, stride)

        super().__init__(out=result, id=(c, k, w, r, stride))


if __name__ == "__main__":
    Conv.main()
