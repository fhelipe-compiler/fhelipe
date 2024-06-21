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

from fheapps.base import Conv, Fc, MaxPool

from .integration_test import IntegrationTest


class TestConv(IntegrationTest):
    make_app = Conv

    def test_small(self):
        self._test(c=1, k=1, w=2, r=1)

    def test_mid(self):
        self._test(c=2, k=4, w=8, r=3)

    def test_strided(self):
        self._test(c=2, k=4, w=8, r=3)


class TestFc(IntegrationTest):
    make_app = Fc

    def test_tiny(self):
        self._test(n=1, m=1)

    def test_normal(self):
        self._test(n=2, m=4)


class TestMaxPool(IntegrationTest):
    make_app = MaxPool

    def test_tiny(self):
        self._test(c=1, w=1, r=1)

    def test_normal(self):
        self._test(c=2, w=4, r=3)
