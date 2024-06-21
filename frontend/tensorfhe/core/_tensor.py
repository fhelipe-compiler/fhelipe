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

from typing import (
    Any,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    final,
    overload,
)

from . import op
from ._vector import OpF, Vector
from .base_op_value import BaseOpValue
from .op import EncOp, Op
from .repack import _manual_repack

Self = TypeVar("Self", bound="Tensor")


class Tensor(Vector):
    def __reshape(self: Self, f: OpF, /, **kwargs) -> Self:
        result = self.apply(f, **kwargs)

        if result.shape != self.shape:
            return result
        else:
            return self

    @overload
    def rotate(self: Self, by: int, /) -> Self:
        ...

    @overload
    def rotate(self: Self, dim: int, by: int) -> Self:
        ...

    def rotate(self: Self, dim: int, by: Optional[int] = None) -> Self:
        if by is None:
            dim, by = 0, dim

        if by == 0:
            return self

        return self.apply(op.Rotate, dim=dim, by=by)

    def shift(self: Self, dim: int, by: int) -> Self:
        if by == 0:
            return self

        return self.apply(op.UnpaddedShift, dim=dim, by=by)

    @_manual_repack
    def _chet_repack(self):
        return self.apply(op.ChetRepack)

    def replicate(self: Self, dim: int, n: int) -> Self:
        return self.insert_dim(dim).__reshape(op.Replicate, dim=dim, n=n)

    def sum(self: Self, dim: int) -> Self:
        return self.__reshape(op.Sum, dim=dim).drop_dim(dim)

    def extend(self: Self, dim: int, size: int) -> Self:
        return self.__reshape(op.Extend, dim=dim, size=size)

    def shrink(self: Self, dim: int, size: int) -> Self:
        return self.__reshape(op.Shrink, dim=dim, size=size)

    def stride(self: Self, dim: int, by: int) -> Self:
        return self.__reshape(op.Stride, dim=dim, by=by)

    def insert_dim(self: Self, dim: int, size: int = 1) -> Self:
        return self.apply(op.InsertDim, dim=dim).extend(dim, size)

    def drop_dim(self: Self, dim: int) -> Self:
        return self.shrink(dim, 1).apply(op.DropDim, dim=dim)

    def reorder_dim(self: Self, order: Sequence[int]) -> Self:
        return self.apply(op.ReorderDim, order=order)

    @property
    def T(self: Self) -> Self:
        return self.reorder_dim((-1, -2))


@final
class OpTensor(BaseOpValue, Tensor):
    def __init__(self, op: Op):
        super().__init__(op)

    @classmethod
    def from_op(cls, op: Op) -> "OpTensor":
        return OpTensor(op)

    @classmethod
    def _apply_bound(cls):
        return Tensor
