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
from abc import abstractmethod
from argparse import ArgumentParser
from typing import Final, Sequence

from ...core import TensorV, stdin, stdout
from ..dataset import DataSet
from ..exe_manager import ExeManager
from .action import Action
from .ds_select import DatasetRangeSelectMixin
from .populate import PopulateDs
from .utils import AvgStats, NnData, prediction_error


class GenTestIn(PopulateDs):
    help = "Populate datasets with test set inputs"

    @abstractmethod
    def test_data(self) -> NnData:
        raise NotImplementedError

    def ds_in(self, _) -> Sequence[TensorV]:
        return [x for x, _ in self.test_data()]


class TestError(DatasetRangeSelectMixin, Action):
    name = "error"
    help = "Check predictions against test targets"

    @abstractmethod
    def test_data(self) -> NnData:
        raise NotImplementedError

    def output_transform(self, x: TensorV) -> TensorV:
        return x

    def __init__(self, output_type: str = "out_tfhe", **kwargs):
        super().__init__(**kwargs)
        self.__output_type: Final = output_type

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        super(TestError, cls).add_cli_args(parser)
        parser.add_argument(
            "output_type",
            choices=["out_unenc", "out_tfhe", "out_torch"],
            help="Selects the outputs for which accuracy is determined",
            nargs="?",
        )

    def run(self, exe: ExeManager) -> float:
        logging.info("Computing test accuracy")
        data = self.test_data()
        err_stats = AvgStats()

        for ds in self.selected_datasets(exe):
            try:
                target = data[int(ds.name)][1]
            except (ValueError, IndexError):
                # This dataset is not part of testing
                continue

            out_path = ds / self.__output_type / stdout
            y = out_path.read_tensor()
            y = self.output_transform(y)

            err = prediction_error(y, target)
            err_stats += err

            if err.num:
                msg = f"WRONG (expected: {target.item()}; got {y.item()})"
            else:
                msg = "ok"
            logging.debug(f"Test {ds.name}: {msg}")

        error = float(err_stats)
        logging.info(f"Test error: {100 * error:.2f}%")
        return error
