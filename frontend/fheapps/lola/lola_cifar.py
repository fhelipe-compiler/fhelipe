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
from fheapps.lola.base_cifar import BaseLolaCifar
from fheapps.lola.shared import pad_pow_2
from tensorfhe import Input, Tensor
from tensorfhe.lib import conv2d_raw, mul_mv


class LolaCifar(BaseLolaCifar[Tensor]):
    def enc(self, x: Input) -> Tensor:
        return pad_pow_2(x).enc()

    def conv(
        self, x: Tensor, w: Input, b: Input, pad: int, out_h: int
    ) -> Tensor:
        w = pad_pow_2(w, dims=slice(2))
        x = conv2d_raw(x, w, stride=(2, 2), padding=(pad, pad))

        b = b.replicate(dim=-1, n=out_h).replicate(dim=-2, n=out_h)
        b = pad_pow_2(b)

        return x + b

    def conv1(self, x: Tensor, w: Input, b: Input) -> Tensor:
        return self.conv(x, w, b, 1, 14)

    def conv2(self, x: Tensor, w: Input, b: Input) -> Tensor:
        return self.conv(x, w, b, 4, 7)

    def fc(self, x: Tensor, w: Input, b: Input) -> Tensor:
        w = pad_pow_2(w)
        b = pad_pow_2(b)
        return mul_mv(w, x)

    def square(self, x: Tensor) -> Tensor:
        return x * x

    def dec(self, x: Tensor) -> Tensor:
        return x


if __name__ == "__main__":
    LolaCifar.main()
