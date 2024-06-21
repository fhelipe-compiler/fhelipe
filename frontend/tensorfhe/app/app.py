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

import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from os import PathLike
from pathlib import Path
from typing import (
    ClassVar,
    Collection,
    Final,
    Generic,
    Mapping,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import __main__

from ..core import Dataflow, Vector, utils
from ..nn import NN
from .actions import (
    ActionT,
    GenericAction,
    NnActionT,
    RunResultT,
    common_actions,
    common_nn_actions,
)
from .exe_manager import ExeManager, NnExeManager

ManagerT = TypeVar("ManagerT")

__IdE = Union[str, int]
InstanceId = Union[__IdE, Sequence[__IdE]]


class BaseApp(ABC, Generic[ManagerT]):
    @classmethod
    def add_instance_cli_args(cls, parser: ArgumentParser) -> None:
        pass

    @staticmethod
    def __base_parser() -> ArgumentParser:
        parser = ArgumentParser(allow_abbrev=False, prefix_chars="-+")

        main_file = Path(__main__.__file__).resolve()
        parser.add_argument(
            "--root",
            type=Path,
            default=main_file.parent / ("_" + main_file.stem),
            dest="root",
            help="The directory in which to put app instances",
        )
        parser.add_argument("-v", "--verbose", action="count", default=0)
        return parser

    @classmethod
    def _argparser(
        cls, actions: Collection[Type[GenericAction[ManagerT]]]
    ) -> ArgumentParser:
        parser = cls.__base_parser()
        cls.add_instance_cli_args(parser)

        sub_commands = parser.add_subparsers(title="Action")
        for a in sorted(actions, key=lambda a: a.name):
            sub_parser = sub_commands.add_parser(a.name, help=a.help)
            sub_parser.set_defaults(action_class=a)
            a.add_cli_args(sub_parser)

        return parser

    @classmethod
    @abstractmethod
    def argparser(cls) -> ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def __init_logging(verbose: int) -> None:
        logging.basicConfig(
            format="{levelname} ({filename}:{lineno}) {message}",
            style="{",
            level=logging.WARNING - 10 * verbose,
        )

    @staticmethod
    def __print_result(result: RunResultT) -> None:
        if result is None:
            return
        elif isinstance(result, (str, int, float, PathLike)):
            print(result)
        else:
            print(*result, sep="\n")

    @classmethod
    def main(cls) -> None:
        parser = cls.argparser()
        raw_args = vars(parser.parse_args())
        kwargs = {k: v for k, v in raw_args.items() if v is not None}

        cls.__init_logging(verbose=kwargs["verbose"])
        instance = cls(**kwargs)

        if "action_class" not in kwargs:
            parser.error("No action specified!")
        action: GenericAction[ManagerT] = kwargs["action_class"](**kwargs)

        exe_manager = instance.exe_manager(kwargs["root"] / instance.id)
        result = action.run(exe_manager)

        cls.__print_result(result)

    def __init__(self, *, id: InstanceId = ""):
        if isinstance(id, (str, int)):
            id = (id,)

        self.id: Final[str] = "_".join(map(str, id))

    @abstractmethod
    def exe_manager(self, root: Path) -> ManagerT:
        raise NotImplementedError


class App(BaseApp[ExeManager]):
    actions: ClassVar[Collection[ActionT]] = common_actions

    @classmethod
    def argparser(cls) -> ArgumentParser:
        return cls._argparser(cls.actions)

    def __init__(
        self, *, out: Union[Vector, Mapping[str, Vector]], id: InstanceId = ""
    ):
        super().__init__(id=id)

        if isinstance(out, Vector):
            out = {utils.stdout: out}

        self.df: Final = Dataflow(out)

    def exe_manager(self, root: Path):
        return ExeManager(self.df, root)


class NnApp(BaseApp[NnExeManager]):
    actions: ClassVar[Collection[NnActionT]] = common_nn_actions

    @classmethod
    def argparser(cls) -> ArgumentParser:
        return cls._argparser(cls.actions)

    def __init__(self, *, nn: NN, id: InstanceId = ""):
        super().__init__(id=id)
        self.nn: Final = nn

    def exe_manager(self, root: Path) -> NnExeManager:
        return NnExeManager(self.nn, root)
