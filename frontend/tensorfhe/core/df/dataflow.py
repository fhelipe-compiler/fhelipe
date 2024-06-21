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

from typing import Final, Iterable, Mapping, TypeVar

import torch

from .._vector import Vector
from ..op import Bootstrap, EncOp, Output
from ..utils import Shape, encode_seq, seq_str
from .traversal_types import *


def encode_op(op: EncOp, ids: IdTraversal) -> Sequence[str]:
    parent_ids = tuple(ids[p] for p in op.parents)
    pt_parents = tuple(i for enc, i in parent_ids if not enc)
    ct_parents = tuple(i for enc, i in parent_ids if enc)

    attr = tuple(op.attributes()) + pt_parents

    # TODO(nsamar): Make this not a hack
    if len(pt_parents) > 0:
        # nsamar: '~' means ContextLogScale
        # user tells backend which which scale inputs and plaintexts have
        # actually want
        attr += ("~",)

    if isinstance(op, Bootstrap):
        # nsamar: user tells backend to which level I bootstrap to
        attr += ("#",)

    # TODO(alex): This is redundant information and should be removed
    parent_shape = op.parents[0].shape

    attr_str = seq_str(attr)
    shape_str = seq_str(encode_seq(parent_shape))
    parents_str = seq_str(encode_seq(ct_parents))

    if len(parent_ids) == 2:
        if pt_parents:
            name = op.name + "CP"
        else:
            name = op.name + "CC"
    else:
        name = op.name

    return (op.name, shape_str, attr_str, parents_str)


def verify_shapes(
    tensors: Mapping[str, TensorV], expected_shapes: Mapping[str, Shape]
) -> None:
    for key, shape in expected_shapes.items():
        if key not in tensors:
            raise ValueError("Missing input", key)
        elif tensors[key].shape != shape:
            raise ValueError(
                f"Unexpected shape for {key}",
                tensors[key].shape,
                "expected",
                shape,
            )


OpT = TypeVar("OpT", bound=Op)


class Dataflow:
    def __init__(self, outputs: Mapping[str, Vector]) -> None:
        self.__out: Final = tuple(
            Output(t.op, name=name) for name, t in outputs.items()
        )

        self.__ids: Final = IdTraversal().traverse(self.__out)
        self.__pt_in: Final = (
            PtInTraversal().traverse(self.__out).unique_values()
        )
        self.__ct_in: Final = (
            CtInTraversal().traverse(self.__out).unique_values()
        )

        self.__in_shapes: Final = dict(
            InputsTraversal().traverse(self.__out).unique_values()
        )
        self.__out_shapes: Final = {o.output_name: o.shape for o in self.__out}

    def encode(self) -> str:
        # alex: `isinstance(op, EncOp)` is here only to make mypy happy.
        cts = (
            op
            for op, (enc, _) in self.__ids.sorted_items()
            if enc and isinstance(op, EncOp)
        )

        rows = tuple(encode_op(ct, self.__ids) for ct in cts)

        columns = zip(*rows)
        col_w = [max(len(s) for s in col) for col in columns]
        col_w[-1] = 0

        return "".join(
            "   ".join(entry.ljust(w) for entry, w in zip(row, col_w)) + "\n"
            for row in rows
        )

    @property
    def in_shapes(self) -> Mapping[str, Shape]:
        return self.__in_shapes

    @property
    def out_shapes(self) -> Mapping[str, Shape]:
        return self.__out_shapes

    def __eval(
        self, ops: Sequence[OpT], inputs: Mapping[str, TensorV]
    ) -> Mapping[OpT, TensorV]:
        verify_shapes(inputs, self.in_shapes)

        with torch.no_grad():
            values = EvaluateTraversal(inputs).traverse(ops)

        return {op: values[op] for op in ops}

    def __ind_eval(
        self, ops: Sequence[Op], inputs: Mapping[str, TensorV]
    ) -> Mapping[str, TensorV]:
        out_dict = self.__eval(ops, inputs)
        return {str(self.__ids[op][1]): v for op, v in out_dict.items()}

    def pt_in(self, inputs) -> Mapping[str, TensorV]:
        return self.__ind_eval(self.__pt_in, inputs)

    def ct_in(self, inputs) -> Mapping[str, TensorV]:
        return self.__ind_eval(self.__ct_in, inputs)

    def outputs(self, inputs) -> Mapping[str, TensorV]:
        out_dict = self.__eval(self.__out, inputs)
        return {op.output_name: v for op, v in out_dict.items()}

    def useful_ops(self) -> int:
        return UsefulOpsTraversal().traverse(self.__out).total()
