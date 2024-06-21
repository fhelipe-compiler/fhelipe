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


class _ScaleTrackingRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("max", torch.ones(()))

    def train(self, mode=True):
        super().train(mode)

        if self.training:
            self.max = torch.ones(())

    def forward(self, x):
        max_x = torch.max(torch.abs(x.detach()))
        self.max = torch.maximum(self.max, max_x)

        return torch.nn.functional.relu(x)


class ApproxRelu(WrapperModule):
    def __init__(self, alpha: int = 14):
        super().__init__(_ScaleTrackingRelu())
        self.alpha: Final = alpha

    def forward_fhe(self, x: FheV) -> FheV:
        scale = self.weights("max", ())
        return lib.relu(x * (1 / scale), alpha=self.alpha) * scale
