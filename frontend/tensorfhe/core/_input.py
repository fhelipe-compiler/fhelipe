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

from typing import Callable, Final, Generic, Type, TypeVar, Union, final

import torch

from . import op
from ._tensor import OpTensor, Tensor
from ._vector import OpVector, Vector
from .base_op_value import BaseOpValue
from .op import Op
from .utils import Index, Shape

Self = TypeVar("Self", bound="Input")
ArgT = Union[Self, float]


class InputPutInstance(Generic[Self]):
    def __init__(self, x: Self, index: Index):
        self.__x: Final = x
        self.__ind: Final = index

    def __call__(self, value: ArgT) -> Self:
        return self.__x.apply(op.SetItem, value, index=self.__ind)


class InputPut(Generic[Self]):
    def __init__(self, x: Self):
        self.__x: Final = x

    def __getitem__(self, index: Index) -> InputPutInstance[Self]:
        return InputPutInstance(self.__x, index)


class Input(Tensor):
    def __truediv__(self: Self, other: ArgT) -> Self:
        return self._apply_uop(op.UFunc, self, other, ufunc=torch.divide)

    def __rtruediv__(self: Self, other: ArgT) -> Self:
        return self._apply_uop(op.UFunc, other, self, ufunc=torch.divide)

    def __pow__(self: Self, other: ArgT) -> Self:
        return self._apply_uop(op.UFunc, self, other, ufunc=torch.pow)

    def __rpow__(self: Self, other: ArgT) -> Self:
        return self._apply_uop(op.UFunc, other, self, ufunc=torch.pow)

    def __getitem__(self: Self, i: Index) -> Self:
        return self.apply(op.GetItem, index=i)

    @property
    def put(self: Self) -> InputPut[Self]:
        return InputPut(self)

    def sqrt(self: Self) -> Self:
        return self.apply(op.UFunc, ufunc=torch.sqrt)

    def reshape(self: Self, shape: Shape) -> Self:
        return self.apply(op.Reshape, shape=shape)

    def flatten(self: Self) -> Self:
        return self.apply(op.Flatten)

    def apply_torch(
        self: Self, module: torch.nn.Module, name: str = ""
    ) -> Self:
        return self.apply(op.TorchOp, module=module, name=name)

    def enc_vector(self) -> Vector:
        return OpVector(self.op.encrypted)

    def enc(self) -> Tensor:
        return OpTensor(self.op.encrypted)


@final
class OpInput(BaseOpValue, Input):
    """
    Implementation of Input.

    Do not construct directly; use methods in __init__.py instead.
    """

    def __init__(self, op: Op):
        if op.is_encrypted:
            raise ValueError(
                "Cannot create Input with already encrypted value", op
            )

        super().__init__(op)

    @classmethod
    def _apply_bound(cls) -> Type:
        return Input
