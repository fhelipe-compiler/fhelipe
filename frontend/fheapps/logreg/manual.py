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

from argparse import ArgumentParser
from typing import Iterator, Sequence, Tuple

import fheapps.logreg.shared as sh
import numpy as np
import tensorfhe as tfhe
import tensorfhe.app.actions as act
from fheapps.logreg.actions import InitLogReg, LogRegAccuracy
from tensorfhe import Input, Vector

VectorN = Sequence[Vector]


class ManualLogReg(tfhe.App):
    actions = {*act.common_actions, InitLogReg, LogRegAccuracy}

    @classmethod
    def add_instance_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("+it", type=int, help="Number of iterations")
        parser.add_argument("+N", type=int, help="Vector size")

    def __shifts(self, n: int) -> Iterator[int]:
        s = 1
        while s < n:
            yield s
            s *= 2

    def __sum_col_vec(self, m: Vector) -> Vector:
        for s in self.__shifts(self.g):
            m = m + m.rotate(-s)

        mask = tfhe.zeros((self.N,)).put[: self.g](1)
        m = m * mask

        for s in self.__shifts(self.g):
            m = m + m.rotate(s)

        return m

    def __sum_row_vec(self, m: Vector) -> Vector:
        for s in self.__shifts(self.m):
            m = m + m.rotate(s * self.g)
        return m

    def __encode_batch(self, b_in: Input) -> VectorN:
        blocks = [
            b_in[:, i : i + self.g].extend(dim=1, size=self.g)
            for i in range(0, b_in.shape[1], self.g)
        ]
        return [b.flatten().enc() for b in blocks]

    def __z(self) -> Sequence[VectorN]:
        batch_shape = (sh.batch_size, sh.feature_cnt)
        batch_in = [
            tfhe.secret_in(str(i), batch_shape) for i in range(sh.batch_cnt)
        ]
        return [self.__encode_batch(b) for b in batch_in]

    def __training_it(
        self, v: VectorN, w: VectorN, z: VectorN, eta: float
    ) -> Tuple[VectorN, VectorN]:
        gamma = sh.lr / self.m

        # 2 levels
        m = sum(self.__sum_col_vec(vi * zi) for vi, zi in zip(v, z))

        # 2 levels (computes `gamma * sig(m) * z`)
        c0, c1, _, c3 = (gamma * c for c in sh.sigmoid_c)

        zc = [tuple(ci * zi for ci in (c0, c1, c3)) for zi in z]
        m2 = m * m
        g = [zc0 + zc1 * m + (zc3 * m) * m2 for zc0, zc1, zc3 in zc]

        # s = c0 + c1 * m + (c3 * m) * (m * m)
        # g = [s * zi for zi in z]

        delta = [self.__sum_row_vec(gi) for gi in g]

        # 1 level
        w_n = [vi + di for vi, di in zip(v, delta)]
        v_n = [(1 - eta) * w_ni + eta * wi for w_ni, wi in zip(w_n, w)]

        return v_n, w_n

    def __compact_out(self, v: VectorN) -> Vector:
        mask = tfhe.zeros((self.N,)).put[: self.g](1)

        masked = [vi * mask for vi in v]

        s = sum(mi.rotate(i * self.g) for i, mi in enumerate(masked))
        assert not isinstance(s, int)
        return s

    def __init__(self, *, it: int = sh.it, N: int = 2**15, **kwargs):
        self.N = N
        self.m = sh.batch_size
        self.g = N // self.m

        z = self.__z()
        zero_vec = tfhe.zeros((N,)).enc_vector()
        v: VectorN = tuple(zero_vec for _ in z[0])
        w = v

        batch_arr = sh.train_batches(z, it=it)
        eta_arr = sh.train_eta(it=it)

        for i, z_b, eta in zip(range(it), batch_arr, eta_arr):
            if i > 0:
                v = [vi._bootstrap() for vi in v]
                w = [wi._bootstrap() for wi in w]

            v, w = self.__training_it(v, w, z_b, eta)

        out = self.__compact_out(v)

        super().__init__(id=(it, "it", N, "N"), out=out)


if __name__ == "__main__":
    ManualLogReg.main()
