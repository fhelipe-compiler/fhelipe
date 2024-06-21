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

"""DSL types for manipulating FHE data.

Types:
    - Vector: (Potentially) encrypted vector (only for low-level code)
    - Tensor: (Potentially) encrypted multi-dimensional tensor
    - Input: Unencrypted input; enables preprocessing data outside of FHE
    - TorchInput: An Input known at compile time (backed by a torch.Tensor)

These types inherit from one another, so for example, an Input can be used anywhere
a Tensor is expected (but not vice-versa).

FHE programs shouldn instantiate these types directly; instead, they should use
`input`, `as_input`, etc. to obtain an Input, and then call `Input.enc()` to obtain
an encrypted Tensor (or `Input.enc_vector()` for a Vector).

Each Input is either _secret_ or _public_: for a secret Input `enc()` produces
an encrypted Tensor; for a public Input it leaves data unencrypted.

Outputs of unary operations take the encrypted/secret/public value of the
inputs; Outputs of multi-argument operations take the maximum value across their
inputs, e.g., summing a secret Input and a public Input produces a secret Input.

Sharing an interface across encrypted/secret/public values allows the same
code to work with different combinaitons of encrypted/secret/public inputs.
Arguments that require unencrypted preporcessing should be declared as Inputs;
the rest should be Tensors.

"""
from typing import TypeVar, Union

from . import op, utils
from ._input import Input, OpInput
from ._tensor import OpTensor, Tensor
from ._torch_input import TorchInput, as_input, ones, zeros
from ._vector import OpVector, Vector
from .df import Dataflow
from .repack import _manual_repack, _result_repack
from .utils import Shape, TensorV, stdin, stdout


def input(name: str, shape: Shape, *, secret: bool) -> Input:
    """Return a handle to a program Input.

    Args:
        name: Filename of the Input
        shape: Tensor shape
        secret: Whether the input is secret or public
    """
    return OpInput(op.Input(name=name, shape=tuple(shape), secret=secret))


def public_in(name: str, shape: Shape) -> Input:
    return input(name, shape, secret=False)


def secret_in(name: str, shape: Shape) -> Input:
    return input(name, shape, secret=True)


def tensor(name: str, shape: Shape) -> Tensor:
    return secret_in(name, shape).enc()


VectorT = TypeVar("VectorT", bound=Vector)
TensorT = TypeVar("TensorT", bound=Tensor)
InputT = TypeVar("InputT", bound=Input)

__all__ = [
    "Dataflow",
    "Input",
    "InputT",
    "Shape",
    "Tensor",
    "TensorT",
    "TensorV",
    "TorchInput",
    "Vector",
    "VectorT",
    "as_input",
    "input",
    "public_in",
    "secret_in",
    "stdin",
    "stdout",
    "tensor",
    "utils",
    "_manual_repack",
    "_result_repack",
    "zeros",
    "ones",
]
