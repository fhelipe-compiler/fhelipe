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

from tensorfhe import TensorT


def pad_pow_2(x: TensorT, dims=slice(None)) -> TensorT:
    shape = list(enumerate(x.shape))[dims]
    for i, d in shape:
        d = 1 << (d - 1).bit_length()
        x = x.extend(dim=i, size=d)
    return x
