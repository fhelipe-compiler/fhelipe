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
from typing import Any, Dict, Final, Sequence, SupportsFloat, Type, TypeVar

import torch

from . import op
from .op import Op
from .utils import TensorV

Self = TypeVar("Self", bound="BaseOpValue")


class BaseOpValue(ABC):
    def __init__(self, op: Op):
        self.__op: Final = op

    @property
    def op(self) -> Op:
        return self.__op

    def from_const(self, v: float):
        v = float(v)

        t = torch.tensor(v, dtype=torch.float64)
        return type(self)(op.Const(tensor=t, clone=False))

    def _apply_raw(self, f: "OpF", args: Sequence["Vector"]):
        if all(isinstance(a, self._apply_bound()) for a in args):
            ops = [a.op for a in args]
            result_op = f(*ops)
            return type(self)(result_op)
        else:
            return NotImplemented

    @classmethod
    @abstractmethod
    def _apply_bound(cls) -> Type:
        raise NotImplementedError


from ._vector import OpF, Vector
