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
from fheapps.lola.base_mnist import BaseLolaMnist
from fheapps.lola.shared import pad_pow_2
from tensorfhe import Input, Tensor
from tensorfhe.lib import conv2d_raw, mul_mv


class LolaMnist(BaseLolaMnist[Tensor]):
    def enc(self, x: Input) -> Tensor:
        return pad_pow_2(x).enc()

    def conv(self, x: Tensor, w: Input, b: Input) -> Tensor:
        w = pad_pow_2(w, dims=slice(1))
        w = w.extend(dim=0, size=8)
        x = conv2d_raw(x, w, stride=(2, 2), padding=(1, 1))

        b = b.reshape(b.shape + (1, 1))
        b = pad_pow_2(b)
        return x + b

    def fc(self, x: Tensor, w: Input, b: Input) -> Tensor:
        w = pad_pow_2(w)
        b = pad_pow_2(b)
        return mul_mv(w, x) + b

    def fc1(self, x: Tensor, w: Input, b: Input) -> Tensor:
        return self.fc(x, w, b)

    def fc2(self, x: Tensor, w: Input, b: Input) -> Tensor:
        return self.fc(x, w, b)


if __name__ == "__main__":
    LolaMnist.main()
