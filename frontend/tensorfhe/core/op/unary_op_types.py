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

from math import prod
from typing import Final, Sequence

import torch

from .. import utils
from ..utils import Attributes, Index, Shape, TensorV
from .op import EncOp, Op


def normalize_dim(shape: Shape, dim: int) -> int:
    if dim < 0:
        out_dim = dim + len(shape)
    else:
        out_dim = dim

    if out_dim < 0 or out_dim >= len(shape):
        raise ValueError("Invalid dimension", dim, shape)
    else:
        return out_dim


def shape_with_dim(shape: Shape, dim: int, size: int) -> Shape:
    return (*shape[:dim], size, *shape[dim + 1 :])


class NoOp(EncOp):
    def __init__(self, par: Op):
        super().__init__(par)

    def evaluate(selt, t):
        return t


class Bootstrap(NoOp):
    name = "BootstrapC"


class ChetRepack(NoOp):
    name = "ChetRepackC"


class BaseShift(EncOp):
    def __init__(self, par: Op, *, dim: int, by: int):
        super().__init__(par)

        self.__dim: Final = normalize_dim(self.shape, dim)

        if abs(by) >= self.shape[self.__dim]:
            raise ValueError("Invalid shift", self.shape, dim, by)
        self.__by: Final = int(by)

    @property
    def wrap_selection(self) -> Index:
        return utils.shift_wrap_selection(self.__dim, self.__by)

    def attributes(self) -> Attributes:
        return (self.__dim, self.__by)

    def evaluate(self, t):
        return t.roll(shifts=self.__by, dims=self.__dim)


class Rotate(BaseShift):
    name = "RotateC"


class UnpaddedShift(BaseShift):
    name = "UnpaddedShiftC"

    def evaluate(self, arr):
        res = super().evaluate(arr)

        # Fill with garbage, unless it's all 0
        if (res[self.wrap_selection] != 0).any():
            res[self.wrap_selection] = 1000

        return res


class VectorRotate(BaseShift):
    name = "HackRotateC"

    def __init__(self, par: Op, *, by: int):
        super().__init__(par, dim=0, by=by)


class ReorderDim(EncOp):
    name = "ReorderDimC"

    def __init__(self, par: Op, *, order: Sequence[int]):
        pad_len = len(par.shape) - len(order)
        if pad_len < 0:
            raise ValueError("Too many dimensions", len(par.shape), order)
        norm_order = (normalize_dim(par.shape, d) for d in order)

        self.__order: Final = (*range(pad_len), *norm_order)
        if not utils.is_a_permutation(self.__order):
            raise ValueError("Not a permutation", len(par.shape), order)

        out_shape = tuple(par.shape[i] for i in self.__order)
        super().__init__(par, shape=out_shape)

    def attributes(self) -> Attributes:
        return utils.encode_seq(self.__order)

    def evaluate(self, t):
        return t.permute(self.__order)


class Stride(EncOp):
    name = "StrideDimC"

    def __init__(self, par: Op, *, dim: int, by: int):
        self.__dim: Final = normalize_dim(par.shape, dim)
        self.__by: Final = int(by)

        if self.__by < 1 or not utils.is_pow_of_2(self.__by):
            raise ValueError("Non-power-of-2 stride", by)

        strided_size = len(range(0, par.shape[self.__dim], self.__by))
        out_shape = shape_with_dim(par.shape, self.__dim, strided_size)

        super().__init__(par, shape=out_shape)

    def attributes(self) -> Attributes:
        return (self.__dim, self.__by)

    def evaluate(self, t):
        s = slice(None, None, self.__by)
        return t[utils.dim_selection(self.__dim, s)]


class ResizeBase(EncOp):
    name = "ResizeDimC"

    def __init__(self, par: Op, *, dim: int, size: int):
        self.__dim: Final = normalize_dim(par.shape, dim)
        self.__size: Final = int(size)

        out_shape = shape_with_dim(par.shape, self.__dim, self.__size)
        super().__init__(par, shape=out_shape)

    def attributes(self) -> Attributes:
        return (self.__dim, self.__size)

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def size(self) -> int:
        return self.__size


class Shrink(ResizeBase):
    def __init__(self, par: Op, *, dim: int, size: int):
        if size > par.shape[dim]:
            raise ValueError("Shrink increases size", par.shape, dim, size)

        super().__init__(par, dim=dim, size=size)

    def evaluate(self, t):
        s = slice(self.size)
        return t[utils.dim_selection(self.dim, s)]


class Extend(ResizeBase):
    def __init__(self, par: Op, *, dim: int, size: int):
        if size < par.shape[dim]:
            raise ValueError("Extend decreases size", par.shape, dim, size)

        super().__init__(par, dim=dim, size=size)

    def evaluate(self, t):
        # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        pad = [0] * (2 * len(self.shape))
        pad[::-2] = (n - o for n, o in zip(self.shape, t.size()))

        return torch.nn.functional.pad(t, pad)


class Sum(EncOp):
    name = "ReduceDimC"

    def __init__(self, par: Op, *, dim: int):
        self.__dim = normalize_dim(par.shape, dim)

        out_shape = shape_with_dim(par.shape, self.__dim, 1)
        super().__init__(par, shape=out_shape)

    def attributes(self) -> Attributes:
        return (self.__dim,)

    def evaluate(self, t):
        return t.sum(dim=self.__dim, keepdims=True)

    def useful_ops(self) -> int:
        return prod(self.parents[0].shape)


class Replicate(EncOp):
    name = "ReplicateDimC"

    def __init__(self, par: Op, *, dim: int, n: int):
        self.__dim: Final = normalize_dim(par.shape, dim)
        self.__n: Final = int(n)

        if par.shape[self.__dim] != 1:
            raise ValueError(
                "Replicate requires a dimension of size 1", par.shape, dim
            )

        out_shape = shape_with_dim(par.shape, self.__dim, self.__n)
        super().__init__(par, shape=out_shape)

    def attributes(self) -> Attributes:
        return (self.__dim, self.__n)

    def evaluate(self, t):
        return t.expand(self.shape)


class DropDim(EncOp):
    name = "DropDimC"

    def __init__(self, par: Op, *, dim: int):
        self.dim: Final = normalize_dim(par.shape, dim)
        if par.shape[self.dim] != 1:
            raise ValueError(
                "DropDim requires a dimension of size 1", par.shape, dim
            )

        out_shape = (*par.shape[: self.dim], *par.shape[self.dim + 1 :])
        super().__init__(par, shape=out_shape)

    def attributes(self) -> Attributes:
        return (self.dim,)

    def evaluate(self, arr):
        return arr.reshape(self.shape)


class InsertDim(EncOp):
    name = "InsertDimC"

    def __init__(self, par: Op, *, dim: int):
        # Like `normalize_dim`, but allows appending of a dimension

        norm_dim = dim
        if norm_dim < 0:
            norm_dim += len(par.shape) + 1
        if norm_dim < 0 or norm_dim > len(par.shape):
            raise ValueError("Invalid dimension", dim, par.shape)

        self.__dim: Final = norm_dim

        out_shape = (*par.shape[: self.__dim], 1, *par.shape[self.__dim :])
        super().__init__(par, shape=out_shape)

    def attributes(self) -> Attributes:
        return (self.__dim,)

    def evaluate(self, t):
        return t.unsqueeze(self.__dim)
