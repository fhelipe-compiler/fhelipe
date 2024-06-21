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
from .module import FheV, SplitModule


class Parameter(SplitModule):
    def __init__(self, data: torch.Tensor):
        super().__init__()

        self.p: Final = torch.nn.parameter.Parameter(data)
        self.__shape: Final = data.size()

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.p

    def forward_fhe(self, x: FheV) -> FheV:
        return self.weights("p", self.__shape)
