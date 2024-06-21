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

from typing import Final

import torch

from .. import lib
from .module import FheV, WrapperModule


class Linear(WrapperModule):
    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__(torch.nn.Linear(in_f, out_f, bias=True))

        self.__in_f: Final = in_f
        self.__out_f: Final = out_f
        self.__bias: Final = bias

    def forward_fhe(self, x: FheV) -> FheV:
        w = self.weights("weight", (self.__out_f, self.__in_f))
        y = lib.mul_mv(w, x)

        if self.__bias:
            b = self.weights("bias", (self.__out_f,))
            y = y + b
        return y
