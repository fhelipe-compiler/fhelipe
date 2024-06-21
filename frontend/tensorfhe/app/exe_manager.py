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

from os import PathLike
from pathlib import Path
from typing import Final

from ..core import Dataflow, secret_in, stdin, stdout
from ..nn import NN
from .datapath import DataPath
from .dataset import DataSet, DataSetMap


class TrainingPath(DataPath):
    def __init__(self, path: PathLike):
        super().__init__(path)

        self.checkpoint: Final = self / "checkpoint.pt"
        self.log: Final = self / "log.txt"


class InstanceRoot(DataPath):
    def __init__(self, path: PathLike):
        super().__init__(path)

        self.datasets: Final = self / "ds"
        self.shared: Final = DataSet(self / "shared")
        self.training: Final = TrainingPath(self / "training")


class ExeManager:
    def __init__(self, df: Dataflow, root: Path):
        self.df: Final = df
        self.root: Final = InstanceRoot(root)
        self.datasets: Final = DataSetMap(self.root.datasets, self.root.shared)


class NnExeManager(ExeManager):
    def __init__(self, nn: NN, root: Path):
        self.nn: Final = nn
        nn.torch.eval()

        x_in = secret_in(stdin, nn.input_shape)
        x_out = nn(x_in)
        df = Dataflow({stdout: x_out})

        super().__init__(df, root)

    def export_weights(self) -> None:
        self.datasets.shared.in_.export_tensors(self.nn.state())

    def load_weights(self) -> None:
        self.nn.load(self.datasets.shared.import_inputs())
