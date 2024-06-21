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

from ..utils import ElementFuncMixin, FheFunctionTestCase


class SignFTest(ElementFuncMixin, FheFunctionTestCase):
    atol = 1e-3

    def gen_tensor(self, rng, shape):
        abs_v = rng.uniform(1.5e-2, 1, shape)
        sign = rng.integers(low=0, high=2, size=shape) * 2 - 1
        return torch.tensor(abs_v * sign)

    def func_fhe(self, t):
        return tfhe.lib.sign(t.enc(), alpha=14)

    def func_clear(self, t):
        return torch.sign(t)


class ReluTest(ElementFuncMixin, FheFunctionTestCase):
    atol = 1e-3

    def func_fhe(self, t):
        return tfhe.lib.relu(t.enc())

    def func_clear(self, t):
        return torch.maximum(t, torch.zeros_like(t))
