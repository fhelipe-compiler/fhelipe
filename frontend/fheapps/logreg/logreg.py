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
from typing import Sequence, Tuple, Union

import fheapps.logreg.shared as sh
import numpy as np
import tensorfhe as tfhe
import tensorfhe.app.actions as act
from fheapps.logreg.actions import InitLogReg, LogRegAccuracy
from tensorfhe import Input, Tensor, _result_repack
from tensorfhe.lib import mul_mv


class LogReg(tfhe.App):
    actions = {
        *act.base_actions,
        act.PopulateSharedNoOp,
        InitLogReg,
        LogRegAccuracy,
    }

    @classmethod
    def add_instance_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("+it", type=int, help="Number of iterations")
        parser.add_argument(
            "+chet",
            action="store_true",
            help="Produce CHET dataflow (duplicate inputs)",
        )

    def __z(self, *, chet: bool) -> Sequence[Tuple[Tensor, Tensor]]:
        batch_shape = (sh.batch_size, sh.feature_cnt)
        z_in = [
            tfhe.secret_in(str(i), batch_shape) for i in range(sh.batch_cnt)
        ]
        if chet:
            return [(z.enc(), z.T.enc()) for z in z_in]
        else:
            return [(z, z.T) for z in (z.enc() for z in z_in)]

    @_result_repack
    def __zt_sig(self, z_t: Tensor, m: Tensor) -> Tensor:
        # Compute `z_t @ (gamma * sig(m))`

        gamma = sh.lr / z_t.shape[1]
        c0, c1, _, c3 = (gamma * c for c in sh.sigmoid_c)

        m = m.replicate(dim=0, n=z_t.shape[0])
        g = (z_t * c0) + (z_t * c1) * m + ((z_t * c3) * m) * (m * m)
        return g.sum(dim=1)

        # Equivalent to:
        # s = c0 + c1 * m + (c3 * m) * (m * m)
        # return mul_mv(z_t, s)

    def __training_it(
        self, v: Tensor, w: Tensor, z: Tensor, z_t: Tensor, eta: float
    ) -> Tuple[Tensor, Tensor]:

        m = mul_mv(z, v)

        delta = self.__zt_sig(z_t, m)

        w_n = v + delta
        v_n = (1 - eta) * w_n + eta * w

        return v_n, w_n

    def __init__(self, *, it: int = sh.it, chet: bool = False, **kwargs):
        v: Tensor = tfhe.as_input(np.zeros(sh.feature_cnt)).enc()
        w = v

        batches_per_it = sh.train_batches(self.__z(chet=chet), it=it)
        eta_per_it = sh.train_eta(it=it)

        for (z, z_t), eta in zip(batches_per_it, eta_per_it):
            v, w = self.__training_it(v, w, z, z_t, eta)

        id: Tuple[Union[int, str], ...] = (it, "it")
        if chet:
            id = ("chet",) + id

        super().__init__(id=id, out=v)


if __name__ == "__main__":
    LogReg.main()
