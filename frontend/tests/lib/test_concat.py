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


class ConcatTest(FheFunctionTestCase):
    def func_fhe(self, x, y, *, dim):
        return tfhe.lib.concat(x, y, dim=dim)

    def func_clear(self, x, y, *, dim):
        return torch.cat((x, y), dim=dim)

    def _test(self, cat_size, *shape, dim=0):
        x_c, y_c = cat_size
        x_s = shape[:dim] + (x_c,) + shape[dim:]
        y_s = shape[:dim] + (y_c,) + shape[dim:]
        self._test_shapes(x_s, y_s, dim=dim)

    def test_1d(self):
        self._test((1, 1))
        self._test((4, 8))
        self._test((6, 3))
        self._test((8, 8))

    def test_higher_dim(self):
        self._test((1, 1), 2, 3)
        self._test((1, 1), 2, 3, dim=2)
        self._test((1, 1), 2, 3, dim=-1)
        self._test((4, 2), 8, 2, dim=1)
        self._test((2, 6), 1, 8, 2, dim=1)

    def test_one_empty(self):
        self._test((0, 0), 0, 0)
        self._test((0, 1), 2, 4)
