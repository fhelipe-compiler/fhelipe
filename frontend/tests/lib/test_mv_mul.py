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

from ..utils import FheFunctionTestCase


class MatrixVectorMulTest(FheFunctionTestCase):
    def func_fhe(self, m, v):
        return tfhe.lib.mul_mv(m.enc(), v.enc())

    def func_clear(self, m, v):
        n = m.shape[0]

        m = m.reshape((n, -1))
        v = v.reshape((-1,))

        return m @ v

    def _test(self, n, m):
        if isinstance(m, int):
            m = (m,)
        self._test_shapes((n, *m), m)

    def test(self):
        self._test(1, 1)
        self._test(1, 5)
        self._test(6, 1)
        self._test(4, 8)
        self._test(2, 5)
        self._test(7, 7)
        self._test(16, 4)

    def test_higher_d(self):
        self._test(1, (1, 1))
        self._test(1, (2, 1))
        self._test(5, (1, 1, 1))
        self._test(3, (4, 1, 2))
