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

import unittest

import tensorfhe as tfhe
import torch
from fheapps.base import max_pool2d

from ..utils import FheFunctionTestCase


class MaxPoolTest(FheFunctionTestCase):
    atol = 5e-3

    def func_fhe(self, t, *, r, s, stride):
        return max_pool2d(t.enc(), (r, s), stride=stride)

    def func_clear(self, t, *, r, s, stride):
        t = torch.as_tensor(t)
        module = torch.nn.MaxPool2d(
            (r, s), stride=stride, padding=(r // 2, s // 2)
        )
        return module(t)

    def _test(self, t_shape, *, r, s=None, stride=1):
        if s is None:
            s = r
        assert r % 2 == s % 2 == 1
        self._test_shapes(t_shape, r=r, s=s, stride=stride)

    def test_one_channel(self):
        self._test((1, 1, 1), r=1)
        self._test((1, 4, 6), r=1)
        self._test((1, 5, 5), r=3)
        self._test((1, 7, 8), r=3, s=1)

    def test_multi_channel(self):
        self._test((5, 1, 1), r=1)
        self._test((4, 6, 10), r=3)
        self._test((9, 3, 7), r=1, s=5)

    def test_strided(self):
        self._test((1, 1, 1), r=1, stride=2)
        self._test((3, 7, 8), r=3, stride=4)
