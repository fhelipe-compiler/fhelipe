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
from functools import cached_property
from typing import Final, Optional, Sequence

import torch

from ..utils import Attributes, Shape, TensorV
from .secrecy import Secrecy


class Op(ABC):
    """
    A node in the operation DAG, for both Input and EncTensor.

    The shape produced by each operation should be known at compile time.
    Any procedure satisfying this can be implemented as an Op; feel free to
    create custom Op subtypes for preprocessing Inputs.

    """

    def __init__(
        self,
        *parents: "Op",
        shape: Optional[Shape] = None,
        secrecy: Secrecy = Secrecy.Public,
    ):
        """
        You shouldn't create Ops by calling their constructor.

        You should either use Input and Tensor member fuctions, or pass your Op
        type to Input.apply() (for custom Op types).
        """
        super().__init__()

        if not all(isinstance(p, Op) for p in parents):
            raise TypeError

        par_secrecy = max((p.secrecy for p in parents), default=Secrecy.Public)
        if par_secrecy == Secrecy.Encrypted:
            parents = tuple(p.encrypted for p in parents)

        self.__parents: Final = tuple(parents)
        self.__secrecy: Final = max(secrecy, par_secrecy)

        if shape is None:
            shape = tuple(self.parents[0].shape)
            if any(p.shape != shape for p in parents):
                raise ValueError([p.shape for p in parents])

        self.__shape: Final[Shape] = shape

    @property
    def shape(self) -> Shape:
        return self.__shape

    @property
    def parents(self) -> Sequence["Op"]:
        return self.__parents

    @property
    def secrecy(self) -> Secrecy:
        return self.__secrecy

    @property
    def is_encrypted(self) -> bool:
        return self.secrecy == Secrecy.Encrypted

    @property
    def is_secret(self) -> bool:
        return self.secrecy >= Secrecy.Secret

    @cached_property
    def encrypted(self) -> "Op":
        if self.secrecy == Secrecy.Secret:
            return Encrypt(self)
        else:
            return self

    @abstractmethod
    def evaluate(self, *parents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def useful_ops(self) -> int:
        return 0


class EncOp(Op):
    """
    An operation the backend can perform on encrypted data.

    You shouldn't subclass this without extending the backend.
    """

    def attributes(self) -> Attributes:
        return tuple()

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


from .special_op_types import Encrypt
