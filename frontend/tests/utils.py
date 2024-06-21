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
import unittest
from abc import ABC, abstractmethod

import numpy as np
import tensorfhe as tfhe
import torch


class DataflowTestCase(unittest.TestCase):
    def setUp(self):
        self.__inputs = {}
        self.__next_in_ind = 0
        self.rng = np.random.default_rng(0)

    def input(self, t, *, secret):
        name = str(self.__next_in_ind)
        self.__next_in_ind += 1

        t = torch.as_tensor(t).detach()
        self.__inputs[name] = t
        return tfhe.input(name, t.size(), secret=secret)

    def eval_v(self, v):
        df = tfhe.Dataflow({"result": v})
        outputs = df.outputs(self.__inputs)
        return outputs["result"]


class FheFunctionTestCase(DataflowTestCase, ABC):
    rtol: float = 1e-05
    atol: float = 0

    @abstractmethod
    def func_fhe(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def func_clear(*args, **kwargs):
        raise NotImplementedError

    def gen_tensor(self, rng, shape):
        return torch.tensor(rng.uniform(-1, 1, shape))

    def secret_configs(self, n):
        all_configs = itertools.product([False, True], repeat=n)
        return filter(any, all_configs)

    def _test_shapes(self, *shapes, **kwargs):
        tensors = [self.gen_tensor(self.rng, s) for s in shapes]
        expected = self.func_clear(*tensors, **kwargs)

        for secret in self.secret_configs(len(shapes)):
            with self.subTest(shapes=shapes, secret=secret, **kwargs):
                tfhe_in = [
                    self.input(t, secret=s) for t, s in zip(tensors, secret)
                ]
                tfhe_out = self.func_fhe(*tfhe_in, **kwargs)
                actual = self.eval_v(tfhe_out)

                np.testing.assert_allclose(
                    actual, expected, atol=self.atol, rtol=self.rtol
                )


class ElementFuncMixin:
    nargs = 1

    def _test(self, s):
        args = [s] * self.nargs
        self._test_shapes(*args)

    def test_0d(self):
        self._test(())

    def test_1d(self):
        self._test((1,))
        self._test((4,))
        self._test((16,))
        self._test((256,))
        self._test((1337,))

    def test_multi_d(self):
        self._test((1, 1))
        self._test((1, 8))
        self._test((8, 4))
        self._test((1, 2, 3, 4, 5, 6))
