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

from typing import Iterable, Sequence

import numpy as np

from ..core import VectorT, _result_repack, as_input
from ..core.utils import dim_selection, shift_wrap_selection


def powers(x: VectorT, max_power: int) -> Sequence[VectorT]:
    power_0 = x.from_const(1).broadcast_to(x.shape)
    powers = [power_0, x]
    for p in range(2, max_power + 1):
        powers.append(powers[p // 2] * powers[p - p // 2])
    return powers


@_result_repack
def poly_eval(x: VectorT, coeffs: Sequence[float]) -> VectorT:
    x_powers = powers(x, len(coeffs) - 1)
    result = sum(p * c for p, c in zip(x_powers, coeffs) if c != 0)

    if isinstance(result, int):  # Empty sum
        return x.from_const(0).broadcast_to(x.shape)
    else:
        return result


def conv_shifts(filter_size: int) -> Iterable[int]:
    start = -(filter_size // 2)
    return reversed(range(start, start + filter_size))
