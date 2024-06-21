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
from argparse import ArgumentParser
from os import PathLike
from typing import ClassVar, Generic, Optional, Sequence, TypeVar, Union

from ..exe_manager import ExeManager, NnExeManager

ManagerT = TypeVar("ManagerT", contravariant=True)

RunResultElement = Union[str, PathLike, int, float]
RunResultT = Union[None, RunResultElement, Sequence[RunResultElement]]


class GenericAction(Generic[ManagerT], ABC):
    name: ClassVar[str]
    help: ClassVar[Optional[str]] = None

    def __init__(self, **kwargs) -> None:
        ...

    @abstractmethod
    def run(self, exe_manager: ManagerT) -> RunResultT:
        raise NotImplementedError

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        pass


Action = GenericAction[ExeManager]
NnAction = GenericAction[NnExeManager]
