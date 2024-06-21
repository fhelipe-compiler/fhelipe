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


class ConvTest(FheFunctionTestCase):
    def func_fhe(self, img_in, wgt_in, *, stride, padding):
        return tfhe.lib.conv2d_raw(
            img_in.enc(), wgt_in, stride=(stride, stride), padding=padding
        )

    def func_clear(self, image, weights, *, stride, padding):
        k, c, r, s = weights.shape
        _, h, w = image.shape

        module = torch.nn.Conv2d(
            c,
            k,
            (r, s),
            stride=stride,
            padding=padding,
            padding_mode="zeros",
            bias=False,
        )
        module.weight = torch.nn.Parameter(weights, requires_grad=False)
        output = module(image)

        _, h_o, w_o = output.shape
        w_pad = (w + stride - 1) // stride - w_o
        h_pad = (h + stride - 1) // stride - h_o
        return torch.nn.functional.pad(output, (0, w_pad, 0, h_pad))

    def _test(self, c, h, w, k, r, s, stride=1):
        for p in range((min(r, s) + 1) // 2):
            self._test_shapes(
                (c, h, w), (k, c, r, s), stride=stride, padding=(p, p)
            )

    def test_one_channel(self):
        self._test(1, 1, 1, 1, 1, 1)
        self._test(1, 4, 2, 1, 1, 1)
        self._test(1, 4, 4, 1, 3, 3)
        self._test(1, 32, 32, 1, 1, 1)
        self._test(1, 32, 32, 1, 5, 5)

    def test_one_to_many(self):
        self._test(1, 1, 1, 5, 1, 1)
        self._test(1, 6, 6, 8, 3, 3)
        self._test(1, 10, 16, 7, 5, 1)

    def test_many_to_one(self):
        self._test(9, 1, 1, 1, 1, 1)
        self._test(6, 4, 8, 1, 3, 5)
        self._test(16, 18, 6, 1, 3, 3)

    def test_many_to_many(self):
        self._test(3, 1, 1, 16, 1, 1)
        self._test(8, 4, 4, 8, 3, 3)
        self._test(4, 4, 8, 6, 3, 1)
        self._test(32, 16, 16, 16, 3, 3)

    def test_strided(self):
        self._test(1, 1, 1, 1, 1, 1, stride=2)
        self._test(1, 4, 4, 1, 1, 1, stride=2)
        self._test(1, 4, 4, 1, 1, 1, stride=4)
        self._test(1, 8, 5, 1, 3, 3, stride=2)
        self._test(4, 7, 11, 6, 3, 5, stride=2)
        self._test(16, 32, 32, 32, 3, 3, stride=2)
