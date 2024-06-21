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

from argparse import ArgumentParser
from typing import Final, Iterator, Optional

from ..dataset import DataSet
from ..exe_manager import ExeManager


class DatasetRangeSelectMixin:
    @staticmethod
    def __lt(x: str, y: str) -> bool:
        try:
            return int(x) < int(y)
        except ValueError:
            return x < y

    @staticmethod
    def __eq(x: str, y: str) -> bool:
        try:
            return int(x) == int(y)
        except ValueError:
            return x == y

    def __init__(
        self,
        *args,
        ds_start: Optional[str] = None,
        ds_end: Optional[str] = None,
        ds_equal: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.__start: Final = ds_start
        self.__end: Final = ds_end
        self.__ds: Final = ds_equal

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        super(DatasetRangeSelectMixin, cls).add_cli_args(parser)  # type: ignore[misc]

        g = parser.add_argument_group(
            title="Dataset selection options",
            description="Options for working with only a subset of the available "
            "datasets. Dataset keys are compared lexicographicaly.",
        )
        g.add_argument(
            "--ds-start",
            "--datasets-start",
            help="Work only on datasets >= DS_START",
        )
        g.add_argument(
            "--ds-end", "--datasets-end", help="Work only on datasets < DS_END"
        )
        g.add_argument(
            "--ds", "--dataset", dest="ds_equal", help="Work only on DS_EQUAL"
        )

    def is_selected(self, name: str) -> bool:
        if self.__start is not None and self.__lt(name, self.__start):
            return False
        elif self.__end is not None and not self.__lt(name, self.__end):
            return False
        elif self.__ds is not None and not self.__eq(name, self.__ds):
            return False
        else:
            return True

    def selected_datasets(self, exe: ExeManager) -> Iterator[DataSet]:
        return (ds for ds in exe.datasets if self.is_selected(ds.name))
