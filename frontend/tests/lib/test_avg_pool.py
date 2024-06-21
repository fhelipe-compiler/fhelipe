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
import torch

from ..utils import FheFunctionTestCase


class AvgPoolTest(FheFunctionTestCase):
    def func_fhe(self, t, *, kernel_size, stride):
        return tfhe.lib.avg_pool2d(
            t.enc(), kernel_size=kernel_size, stride=stride
        )

    def func_clear(self, t, *, kernel_size, stride):
        return torch.nn.functional.avg_pool2d(
            t,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )

    def _test(self, *dims, kernel_size=2, stride=1):
        assert kernel_size % 2 == 1 or stride % 2 == 0
        self._test_shapes(dims, kernel_size=kernel_size, stride=stride)

    def test_one_channel(self):
        self._test(1, 1, 1, kernel_size=1, stride=1)
        self._test(1, 2, 2, kernel_size=1, stride=1)
        self._test(1, 2, 2, kernel_size=3, stride=1)
        self._test(1, 2, 2, kernel_size=2, stride=2)
        self._test(1, 2, 2, kernel_size=3, stride=2)
        self._test(1, 4, 4, kernel_size=3, stride=1)
        self._test(1, 4, 4, kernel_size=2, stride=2)
        self._test(1, 16, 12, kernel_size=5, stride=2)
        self._test(1, 16, 12, kernel_size=4, stride=2)
        self._test(1, 4, 6, kernel_size=3, stride=1)

    def test_many_channels(self):
        self._test(5, 1, 1, kernel_size=1, stride=1)
        self._test(2, 4, 6, kernel_size=3, stride=1)
        self._test(8, 4, 6, kernel_size=3, stride=1)
        self._test(8, 4, 6, kernel_size=2, stride=2)
        self._test(8, 8, 6, kernel_size=3, stride=2)

    def test_many_dims(self):
        self._test(5, 3, 6, 8, kernel_size=3, stride=2)
