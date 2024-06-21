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

from abc import ABC, abstractmethod
from functools import partial
from math import prod
from typing import (
    Any,
    List,
    Mapping,
    Protocol,
    Sequence,
    SupportsFloat,
    Type,
    TypeVar,
    Union,
    final,
)

import torch

from . import op
from .op import Op, Secrecy
from .repack import auto_repack_enabled
from .utils import Shape, TensorV, is_pow_of_2

Self = TypeVar("Self", bound="Vector")
OtherT = Union[Self, float]


class OpF(Protocol):
    def __call__(self, *ops: Op, **kwargs) -> Op:
        ...


class Vector(ABC):
    @abstractmethod
    def from_const(self: Self, v: float) -> Self:
        raise NotImplementedError

    @property
    @abstractmethod
    def op(self) -> Op:
        raise NotImplementedError

    @abstractmethod
    def _apply_raw(self: Self, op_f: OpF, args: Sequence[Self]) -> Self:
        raise NotImplementedError

    def _apply_self_t(
        self: Self,
        f: OpF,
        raw_args: Sequence[OtherT],
        broadcast: bool = False,
    ) -> Self:
        args: List[Any] = [
            a if isinstance(a, Vector) else self.from_const(a) for a in raw_args
        ]

        if broadcast:
            arg_shapes = (a.shape for a in args)
            out_shape = torch.broadcast_shapes(*arg_shapes)
            args = [a.broadcast_to(out_shape) for a in args]

        result = self._apply_raw(f, args)
        if result is NotImplemented:
            return NotImplemented

        if auto_repack_enabled():
            result = result._chet_repack()
        return result

    @staticmethod
    def _apply(
        f: OpF,
        raw_args: Sequence[Union["Vector", float]],
        kwargs: Mapping[str, Any] = {},
        broadcast: bool = False,
    ) -> Any:
        f = partial(f, **kwargs)
        for a in raw_args:
            if not isinstance(a, Vector):
                continue

            result = a._apply_self_t(f, raw_args, broadcast=broadcast)
            if result is NotImplemented:
                continue

            return result

        raise TypeError

    def _apply_uop(self: Self, f: OpF, /, *args: OtherT, **kwargs) -> Self:
        try:
            return self._apply(f, args, kwargs, broadcast=True)
        except TypeError as err:
            return NotImplemented

    def apply(self: Self, f: OpF, /, *others: OtherT, **kwargs) -> Self:
        return self._apply(f, (self, *others), kwargs)

    def apply_broadcast(
        self: Self, f: OpF, /, *others: OtherT, **kwargs
    ) -> Self:
        return Vector._apply(f, (self, *others), kwargs, broadcast=True)

    @property
    def shape(self) -> Shape:
        return self.op.shape

    def __len__(self) -> int:
        return prod(self.shape)

    def rotate(self: Self, by: int, /) -> Self:
        if by == 0:
            return self

        return self.apply(op.VectorRotate, by=by)

    def _bootstrap(self: Self) -> Self:
        return self.apply(op.Bootstrap)

    def _chet_repack(self: Self) -> Self:
        return self

    def broadcast_to(self: Self, shape: Shape) -> Self:
        """
        Raises:
            ValueError: If `self` is encrypted and it's not shaped like `shape`.
        """
        if shape == self.shape:
            return self
        else:
            return self.apply(op.Broadcast, shape=shape)

    def __add__(self: Self, other: OtherT) -> Self:
        return self._apply_uop(op.Add, self, other)

    def __radd__(self: Self, other: OtherT) -> Self:
        return self._apply_uop(op.Add, other, self)

    def __mul__(self: Self, other: OtherT) -> Self:
        return self._apply_uop(op.Mul, self, other)

    def __rmul__(self: Self, other: OtherT) -> Self:
        return self._apply_uop(op.Mul, self, other)

    def __sub__(self: Self, other: OtherT) -> Self:
        return self + (-1 * other)

    def __rsub__(self: Self, other: OtherT) -> Self:
        return other + (-1 * self)


from .base_op_value import BaseOpValue


@final
class OpVector(BaseOpValue, Vector):
    def __init__(self, op: Op):
        super().__init__(op)

        if len(self.shape) > 1:
            raise ValueError(
                "Vectors must have at most 1 dimension", self.shape
            )
        if not is_pow_of_2(len(self)):
            raise ValueError("Vector size must be a power of 2", self.shape)

    @classmethod
    def _apply_bound(cls) -> Type:
        return Vector
