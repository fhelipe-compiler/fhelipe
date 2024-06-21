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

import torch
from fheapps.tensor import fft_dif, inner_prod, mttkrp, mul_tm, mul_tv

from ..utils import FheFunctionTestCase


class MulTVTest(FheFunctionTestCase):
    def func_fhe(self, t, v):
        return mul_tv(t.enc(), v.enc())

    def func_clear(self, t, v):
        return t @ v

    def _test(self, i, j, k):
        self._test_shapes((i, j, k), (k,))

    def test_tiny(self):
        self._test(1, 1, 1)

    def test_normal(self):
        self._test(3, 4, 2)
        self._test(4, 5, 7)


class MulTMTest(FheFunctionTestCase):
    def func_fhe(self, t, m):
        return mul_tm(t.enc(), m.enc())

    def func_clear(self, t, m):
        return t @ m.T

    def _test(self, i, j, k, l):
        self._test_shapes((i, j, l), (k, l))

    def test_tiny(self):
        self._test(1, 1, 1, 1)

    def test_normal(self):
        self._test(2, 3, 4, 5)
        self._test(5, 7, 8, 6)


class MTTKRPTest(FheFunctionTestCase):
    def func_fhe(self, b, c, d):
        return mttkrp(b.enc(), c.enc(), d.enc())

    def func_clear(self, b, c, d):
        I, L, K = b.shape
        J, _ = c.shape

        a = torch.zeros((I, J))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    for l in range(L):
                        a[i, j] += b[i, l, k] * c[j, k] * d[l, j]
        return a

    def _test(self, i, j, k, l):
        self._test_shapes((i, l, k), (j, k), (l, j))

    def test_tiny(self):
        self._test(1, 1, 1, 1)

    def test_normal(self):
        self._test(2, 5, 3, 4)

    def test_reordered(self):
        self._test(6, 2, 4, 3)


class InnerProdTest(FheFunctionTestCase):
    def func_fhe(self, a, b):
        return inner_prod(a.enc(), b.enc())

    def func_clear(self, a, b):
        return torch.sum(a * b)

    def _test(self, *shape):
        self._test_shapes(shape, shape)

    def test_tiny(self):
        self._test()
        self._test(1)
        self._test(1, 1)

    def test_1d(self):
        self._test(5)

    def test_2d(self):
        self._test(4, 6)
        self._test(8, 2)

    def test_multi_d(self):
        self._test(3, 1, 4)
        self._test(1, 5, 9, 2)
        self._test(1, 1, 1, 1, 1, 1, 2, 1)


class FFTTest(FheFunctionTestCase):
    def func_fhe(self, v, real):
        c = fft_dif(v)
        return c.r if real else c.i

    @staticmethod
    def __bit_reverse(i: int, n: int) -> int:
        j = 0
        while n > 1:
            j = j * 2 + i % 2
            i //= 2
            n //= 2
        return j

    def func_clear(self, v, real):
        raw_fft = torch.fft.fft(v).conj()

        n = len(v)
        out = torch.tensor(
            [raw_fft[self.__bit_reverse(i, n)] for i in range(n)]
        )
        return out.real if real else out.imag

    def _test(self, n):
        self._test_shapes((n,), real=True)
        self._test_shapes((n,), real=False)

    def test_small(self):
        self._test(1)
        self._test(2)
        self._test(8)

    def test_moderate(self):
        self._test(2**10)
