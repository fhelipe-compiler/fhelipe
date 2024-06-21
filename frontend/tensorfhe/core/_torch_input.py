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

from typing import Final, Sequence, Union, final, overload

import torch
from numpy.typing import ArrayLike

from . import op
from ._input import Input
from ._vector import OpF, Vector
from .utils import Shape, TensorV

InputLike = Union[ArrayLike, torch.Tensor]


@final
class TorchInput(Input):
    def __init__(self, value: torch.Tensor) -> None:
        self.__t: Final = value

    @property
    def tensor(self) -> TensorV:
        return self.__t

    @property
    def op(self) -> op.Const:
        return op.Const(tensor=self.__t, clone=False)

    def from_const(self, v: float) -> "TorchInput":
        t = torch.as_tensor(
            v, dtype=self.tensor.dtype, device=self.tensor.device
        )
        return TorchInput(t)

    def _apply_raw(self, f: OpF, args: Sequence) -> "TorchInput":
        if all(isinstance(a, TorchInput) for a in args):
            ops = [a.op for a in args]
            result_op = f(*ops)

            tensors = [a.tensor for a in args]
            result_tensor = result_op.evaluate(*tensors)

            return TorchInput(result_tensor)
        else:
            return NotImplemented

    def enc(self) -> "TorchInput":
        return self


@overload
def as_input(value: Union[TorchInput, InputLike]) -> TorchInput:
    ...


@overload
def as_input(value: Input) -> Input:
    ...


def as_input(value: Union[InputLike, Input]) -> Input:
    """Convert value to an Input.

    Args:
        value: Either an Input or something that can be converted to a
            torch.Tensor.
    Returns:
        value, if already an Input; otherwise, value converted to a TorchInput.

    """
    if isinstance(value, Input):
        return value
    else:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value, dtype=torch.float64)

        return TorchInput(tensor)


def zeros(shape: Shape) -> TorchInput:
    return TorchInput(torch.zeros(shape, dtype=torch.float64))


def ones(shape: Shape) -> TorchInput:
    return TorchInput(torch.ones(shape, dtype=torch.float64))
