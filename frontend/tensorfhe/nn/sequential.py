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

from .module import FheV, WrapperModule


class Sequential(WrapperModule):
    def __init__(self, *modules: WrapperModule):
        torch_modules = (m.torch for m in modules)
        super().__init__(torch.nn.Sequential(*torch_modules))

        self.__modules: Final = tuple(modules)
        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def forward_fhe(self, x: FheV) -> FheV:
        for m in self.__modules:
            x = m(x)
        return x
