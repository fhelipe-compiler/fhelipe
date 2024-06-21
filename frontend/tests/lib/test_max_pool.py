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

import itertools

import tensorfhe as tfhe
import torch

from ..utils import FheFunctionTestCase


class MaximumTest(FheFunctionTestCase):
    atol = 5e-3

    def func_fhe(self, *t_in):
        t = [t.enc() for t in t_in]
        res = tfhe.lib.maximum(t)
        return res

    def secret_configs(self, n):
        if n < 4:
            return super().secret_configs(n)
        else:
            all_true = [True] * n
            rand_configs = (
                self.rng.choice([True, False], size=n, replace=True).tolist()
                for _ in range(10)
            )
            all_configs = itertools.chain((all_true,), rand_configs)

        return filter(any, all_configs)

    def func_clear(self, *tensors):
        res = tensors[0]
        for t in tensors:
            res = torch.maximum(res, t)
        return res

    def _test(self, shape, n=2):
        shapes = [shape] * n
        self._test_shapes(*shapes)

    def test_0d(self):
        self._test(())
        self._test((), n=3)

    def test_1d(self):
        self._test((1,))
        self._test((5,))
        self._test((17,))

    def test_multi_d(self):
        self._test((1, 1, 1))
        self._test((4, 3, 8))

    def test_many(self):
        self._test((1, 1, 1, 1), n=15)
        self._test((1, 1, 1), n=3)
        self._test((1, 1, 1, 1), n=7)
        self._test((1, 1), n=5)
