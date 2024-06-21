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
from math import prod
from typing import Final

from .op import EncOp


class BinaryOpBase(EncOp, ABC):
    def __init__(self, p1, p2):
        super().__init__(p1, p2)

    def useful_ops(self) -> int:
        return prod(self.shape)

    @property
    @abstractmethod
    def base_name(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        c_str = "".join("C" for p in self.parents if p.is_encrypted)
        p_str = "".join("P" for p in self.parents if not p.is_encrypted)
        return self.base_name + c_str + p_str


class Add(BinaryOpBase):
    base_name: Final = "Add"

    def evaluate(self, t1, t2):
        return t1 + t2


class Mul(BinaryOpBase):
    base_name: Final = "Mul"

    def evaluate(self, t1, t2):
        return t1 * t2
