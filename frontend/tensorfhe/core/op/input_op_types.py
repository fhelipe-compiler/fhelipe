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

import math
from typing import Any, Callable, Final, List, Optional, Sequence, Tuple, Union

import torch

from ..utils import Index, IndexElement, Shape, TensorV
from .op import Op
from .special_op_types import Input


class InputOp(Op):
    def __init__(self, *parents: Op, shape: Optional[Shape] = None):
        super().__init__(*parents, shape=shape)
        if self.is_encrypted:
            raise ValueError(
                "Operation does not support encrypted data", type(self)
            )


class Reshape(InputOp):
    def __init__(self, par: Op, *, shape: Shape):
        if math.prod(par.shape) != math.prod(shape):
            raise ValueError(
                "Reshape cannot change number of tensor elements",
                par.shape,
                shape,
            )

        super().__init__(par, shape=shape)

    def evaluate(self, t):
        return t.reshape(self.shape)


class Broadcast(InputOp):
    def __init__(self, par: Op, *, shape: Shape):
        if torch.broadcast_shapes(par.shape, shape) != shape:
            raise ValueError("Cannot broadcast", par.shape, shape)

        super().__init__(par, shape=shape)

    def evaluate(self, t):
        return t.expand(self.shape)


def indexed_shape(shape: Shape, index: Index) -> Shape:
    seq_index: Tuple[Any, ...]

    if isinstance(index, Sequence):
        seq_index = tuple(index)
    else:
        seq_index = (index,)

    if seq_index.count(Ellipsis) > 1:
        raise ValueError("Invalid index", shape, index)
    if seq_index.count(Ellipsis) == 0:
        seq_index = seq_index + (Ellipsis,)

    if len(seq_index) > len(shape) + 1:
        raise ValueError("Invalid index", shape, index)

    pad_cnt = len(shape) - (len(seq_index) - 1)
    pad_i = seq_index.index(Ellipsis)

    seq_index = (
        seq_index[:pad_i] + (slice(None),) * pad_cnt + seq_index[pad_i + 1 :]
    )

    result: List[int] = []

    for dim, i in zip(shape, seq_index):
        if isinstance(i, slice):
            selected = range(dim)[i]
            result.append(len(selected))

    return tuple(result)


class GetItem(InputOp):
    def __init__(self, par: Op, *, index: Index):
        self.__index: Final = index

        shape = indexed_shape(par.shape, index)
        super().__init__(par, shape=shape)

    def evaluate(self, t):
        return t[self.__index]


class SetItem(InputOp):
    def __init__(self, lhs: Op, rhs: Op, *, index: Index):
        self.__index: Final = index

        ind_shape = indexed_shape(lhs.shape, index)
        if torch.broadcast_shapes(ind_shape, rhs.shape) != ind_shape:
            raise ValueError("Cannot broadcast", rhs.shape, ind_shape)

        super().__init__(lhs, rhs, shape=lhs.shape)

    def evaluate(self, lhs, rhs):
        result = lhs.clone()
        result[self.__index] = rhs
        return result


class UFunc(InputOp):
    def __init__(self, *pars: Op, ufunc: Callable[..., TensorV]):
        self.__f: Final = ufunc

        par_shapes = (p.shape for p in pars)
        shape = torch.broadcast_shapes(*par_shapes)
        super().__init__(*pars, shape=shape)

    def evaluate(self, *t):
        return self.__f(*t)


class Flatten(InputOp):
    def __init__(self, par: Op):
        n = math.prod(par.shape)
        super().__init__(par, shape=(n,))

    def evaluate(self, par):
        return par.flatten()


class TorchOp(InputOp):
    def __init__(self, x: Op, *, module: torch.nn.Module, name: str = ""):
        self.__module = module

        prefix = name + "." if name else ""
        w_ops = tuple(
            Input(name=prefix + key, shape=w.shape, secret=False)
            for key, w in module.state_dict().items()
        )

        dummy_in = torch.tensor(0).expand(x.shape)
        output_shape = module(dummy_in).size()

        super().__init__(x, *w_ops, shape=output_shape)

    def evaluate(self, x, *weights):
        state_dict = {
            n: w for n, w in zip(self.__module.state_dict().keys(), weights)
        }
        self.__module.load_state_dict(state_dict)
        return self.__module(x)
