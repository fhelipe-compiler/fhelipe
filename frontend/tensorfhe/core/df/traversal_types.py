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

from typing import Final, Mapping, Sequence, Tuple

from ..op import Encrypt, Input, Op
from ..utils import Shape, TensorV
from .op_traversal import GcOpTraversal, OpMultiTraversal, OpTraversal


class IdTraversal(OpTraversal[Tuple[bool, int]]):
    def __init__(self) -> None:
        super().__init__()
        # Encoding requires encrypted values to be indexed separately
        self.__next_i: Final = {True: 0, False: 0}

    def _evaluate(self, op: Op, _) -> Tuple[bool, int]:
        i = self.__next_i[op.is_encrypted]
        self.__next_i[op.is_encrypted] += 1
        return (op.is_encrypted, i)


class InputsTraversal(OpMultiTraversal[Tuple[str, Shape]]):
    def _evaluate(self, op, _) -> Sequence[Tuple[str, Shape]]:
        if isinstance(op, Input):
            return ((op.input_name, op.shape),)
        else:
            return ()


class CtInTraversal(OpMultiTraversal[Op]):
    def _evaluate(self, op: Op, _) -> Sequence[Op]:
        if isinstance(op, Encrypt):
            return op.parents
        else:
            return ()


class PtInTraversal(OpMultiTraversal[Op]):
    def _evaluate(self, op: Op, _) -> Sequence[Op]:
        if op.is_encrypted and not isinstance(op, Encrypt):
            return tuple(p for p in op.parents if not p.is_encrypted)
        else:
            return ()


class EvaluateTraversal(GcOpTraversal[TensorV]):
    def __init__(self, inputs: Mapping[str, TensorV]) -> None:
        super().__init__()
        self.__inputs: Final = inputs

    def _evaluate(self, op: Op, parents: Sequence[TensorV]) -> TensorV:
        if isinstance(op, Input):
            result = self.__inputs[op.input_name]
        else:
            result = op.evaluate(*parents)

        if result.shape != op.shape:
            raise RuntimeError(
                f"Unexpected array shape! Expected {op.shape}; got {result.shape}"
            )

        return result


class UsefulOpsTraversal(OpTraversal[int]):
    def _evaluate(self, op: Op, _) -> int:
        return op.useful_ops() if op.encrypted else 0

    def total(self):
        return sum(self.values())
