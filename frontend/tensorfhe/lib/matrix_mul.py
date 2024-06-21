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

from ..core import TensorT, _result_repack


@_result_repack
def mul_mv(matrix: TensorT, vector: TensorT) -> TensorT:
    """
    Multilpy an N x M matrix with an M vector.

    M can consist of multiple dimensions.
    """
    n = matrix.shape[0]

    if vector.shape != matrix.shape[1:]:
        raise ValueError("Invalid vector shape", vector.shape, matrix.shape)

    replicated_v = vector.replicate(dim=0, n=n)
    products = replicated_v * matrix

    while len(products.shape) > 1:
        products = products.sum(dim=-1)

    return products
