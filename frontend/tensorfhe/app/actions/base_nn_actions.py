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
from os import PathLike
from typing import Final, Mapping, Optional

import torch

from ...core import TensorV, stdin, stdout
from ..datapath import DataPath
from ..exe_manager import ExeManager, NnExeManager
from .action import Action, NnAction
from .ds_select import DatasetRangeSelectMixin
from .populate import PopulateShared, _PopulateShared
from .train import Checkpoint


class GenOutTorch(DatasetRangeSelectMixin, NnAction):
    name = "out-torch"
    help = "Generate outputs using PyTorch in `out_torch/*`"

    def run(self, exe: NnExeManager) -> None:
        exe.load_weights()
        exe.nn.torch.eval()

        for ds in self.selected_datasets(exe):
            out_torch = ds.out_torch
            logging.info(f"Generating `{out_torch.name}` in {ds.name}")

            x = (ds.in_ / stdin).read_tensor().to(dtype=torch.float)
            x = x.unsqueeze(0)  # Add a batch dimension

            with torch.no_grad():
                y = exe.nn.torch(x)

            y = y.squeeze(0)
            (out_torch / stdout).write_tensor(y)


class GenRandWeights(NnAction):
    name = "rand-weights"
    help = "Generate random NN weights"

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.__seed: Final = seed

    @classmethod
    def add_cli_args(cli, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--seed", help="The seed for generating random tensors"
        )

    def run(self, exe: NnExeManager) -> None:
        if self.__seed:
            torch.manual_seed(self.__seed)
            for c in exe.nn.torch.modules():
                try:
                    c.reset_parameters()  # type: ignore[operator]
                except AttributeError:
                    pass

        exe.export_weights()


class ClearTraining(Action):
    name = "clear-training"
    help = "Delete checkpoint and logs created by `train`"

    def run(self, exe: ExeManager) -> None:
        logging.info("Deleting training checkpoint.")
        exe.root.training.checkpoint.unlink()
        exe.root.training.log.unlink()


class InstallWeights(_PopulateShared):
    name = "install-weights"
    help = "Install weights from active training checkpoint"
    checkpoint: Optional[PathLike] = None

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--last",
            action="store_true",
            dest="install_last",
            help="Install weights from last training epoch."
            "(otherwise, use the epoch with the lowest validation loss)",
        )

    def __init__(self, install_last: bool = False, **kwargs):
        self.__install_last: Final = install_last

    def shared_in(self, exe: ExeManager) -> Mapping[str, TensorV]:
        if self.checkpoint is not None:
            checkpoint_path = exe.root / DataPath(self.checkpoint)
        else:
            checkpoint_path = exe.root.training.checkpoint

        cp = Checkpoint.load(checkpoint_path)
        return cp.model if self.__install_last else cp.best_model


class PopulateSharedCheckpoint(InstallWeights):
    name = PopulateShared.name
    help = "Initialize with pre-trained weights"
