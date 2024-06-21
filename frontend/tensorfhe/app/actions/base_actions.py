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
from argparse import ArgumentParser
from typing import Final

from ..datapath import DataPath
from ..dataset import DataSet
from ..exe_manager import ExeManager
from .action import Action
from .ds_select import DatasetRangeSelectMixin


class GenDataflow(Action):
    name = "tdf"
    help = "Generate tensor dataflow"

    def run(self, exe: ExeManager) -> DataPath:
        tdf = exe.datasets.shared.tdf
        logging.info(f"Generating dataflow in `{tdf.name}`")
        tdf.write_text(exe.df.encode())
        return exe.root


class CountUsefulOps(Action):
    name = "useful-ops"
    help = "Count the number of useful scalar ops"

    def run(self, exe: ExeManager) -> int:
        return exe.df.useful_ops()


class InitDs(Action):
    name = "init-ds"
    help = "Initialize a bare dataset with all necessary links"

    def __init__(self, dataset: str, **kwargs):
        self.__dataset: Final = dataset

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        super(InitDs, cls).add_cli_args(parser)
        parser.add_argument("dataset")

    def run(self, exe: ExeManager) -> DataSet:
        logging.info(f"Initializing dataset `{self.__dataset}`")
        ds = exe.datasets[self.__dataset]
        return ds


class InitShared(Action):
    name = "init-shared"
    help = "Initialize a bare shared directory"

    def run(self, exe: ExeManager) -> DataPath:
        return exe.root.shared.mkdir()


class BackendIn(DatasetRangeSelectMixin, Action):
    name = "backend-in"
    help = "Use application inputs to produce the inputs for the backend"

    def run(self, exe: ExeManager) -> None:
        for ds in self.selected_datasets(exe):
            logging.info(
                f"Generating `{ds.pt_in.name}` and `{ds.ct_in.name}` in {ds.name}"
            )
            inputs = ds.import_inputs()

            ds.pt_in.export_tensors(exe.df.pt_in(inputs))
            ds.ct_in.export_tensors(exe.df.ct_in(inputs))


class GenOutTfhe(DatasetRangeSelectMixin, Action):
    name = "out-tfhe"
    help = "Generate TFHE verification outputs"

    def run(self, exe: ExeManager) -> None:
        for ds in self.selected_datasets(exe):
            out_tfhe = ds.out_tfhe
            logging.info(f"Generating `{out_tfhe.name}` in {ds.name}")

            inputs = ds.import_inputs()
            outputs = exe.df.outputs(inputs)
            out_tfhe.export_tensors(outputs)
