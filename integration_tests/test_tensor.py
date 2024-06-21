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

from fheapps.tensor import FFT, MTTKRP, TTM

from .integration_test import IntegrationTest


class TestFFT(IntegrationTest):
    make_app = FFT

    def test_default(self):
        self._test_frontend()


class TestMTTKRP(IntegrationTest):
    make_app = MTTKRP

    def test_default(self):
        self._test_frontend()


class TestTTM(IntegrationTest):
    make_app = TTM

    def test_default(self):
        self._test_frontend()
