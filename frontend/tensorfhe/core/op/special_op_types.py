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

from typing import Final, Sequence

import numpy as np
import torch

from ..utils import Attributes, Shape, TensorV, is_a_valid_name
from .op import EncOp, Op
from .secrecy import Secrecy


class Input(Op):
    def __init__(self, *, name: str, shape: Shape, secret: bool):
        secrecy = Secrecy.Secret if secret else Secrecy.Public

        super().__init__(shape=shape, secrecy=secrecy)

        if not is_a_valid_name(name):
            raise ValueError("Invalid input name", name)
        self.__name: Final = name

    @property
    def input_name(self) -> str:
        return self.__name

    def evaluate(self):
        raise TypeError


class Const(Op):
    def __init__(self, *, tensor: TensorV, clone=True, detach=False):
        self.__t = tensor

        if detach:
            self.__t = self.__t.detach()
        if clone:
            self.__t = self.__t.clone()
        super().__init__(shape=tuple(self.__t.size()))

    def evaluate(self):
        return self.t

    @property
    def t(self) -> TensorV:
        return self.__t


class Encrypt(EncOp):
    name = "InputC"

    def __init__(self, par: Op):
        if par.secrecy != Secrecy.Secret:
            raise ValueError("Only secret values can be encrypted", par.secrecy)

        super().__init__(par, secrecy=Secrecy.Encrypted)

    def evaluate(self, t):
        return t


class Output(EncOp):
    name = "OutputC"

    def __init__(self, par: Op, *, name: str):
        if not is_a_valid_name(name):
            raise ValueError("Invalid output name", name)
        if not par.encrypted:
            raise ValueError("Unencrypted program output", name)

        self.output_name = name
        super().__init__(par)

    def attributes(self) -> Attributes:
        return (self.output_name,)

    def evaluate(self, t):
        return t
