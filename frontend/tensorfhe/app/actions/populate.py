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
from os import PathLike
from typing import Mapping, Sequence, TypeVar, Union

import torch

from ...core import TensorV, stdin
from ..datapath import Id
from ..dataset import DataSet
from ..exe_manager import ExeManager
from .action import Action, GenericAction
from .ds_select import DatasetRangeSelectMixin

ManagerT = TypeVar("ManagerT", bound=ExeManager)


class _PopulateShared(GenericAction[ManagerT]):
    help = "Create shared input files"

    @abstractmethod
    def shared_in(self, exe: ManagerT) -> Mapping[str, TensorV]:
        raise NotImplementedError

    def run(self, exe: ManagerT) -> DataSet:
        tensors = self.shared_in(exe)
        exe.root.shared.in_.export_tensors(tensors)
        return exe.root.shared


class PopulateShared(_PopulateShared[ManagerT]):
    name = "in-shared"


class PopulateSharedNoOp(PopulateShared[ExeManager]):
    help = f"{PopulateShared.help} (this app has none)"

    def shared_in(self, exe: ExeManager) -> Mapping[str, TensorV]:
        return {}


DsIn = Union[Mapping[str, TensorV], TensorV]


class PopulateDs(DatasetRangeSelectMixin, Action):
    name = "in-ds"
    help = "Create input files for datasets"

    @abstractmethod
    def ds_in(self, exe: ExeManager) -> Sequence[DsIn]:
        raise NotImplementedError

    def run(self, exe: ExeManager) -> Sequence[DataSet]:
        logging.info(f"Setting up dataset inputs")
        data = self.ds_in(exe)

        max_i = len(data) - 1
        padding = len(str(max_i))

        datasets = []
        for i, ds_in in enumerate(data):
            key = str(i).zfill(padding)

            if not self.is_selected(key):
                continue

            logging.debug(f"Inputs for dataset `{key}`")
            ds = exe.datasets[key]

            if isinstance(ds_in, TensorV):
                (ds.in_ / stdin).write_tensor(ds_in)
            else:
                ds.in_.export_tensors(ds_in)

            datasets.append(ds)

        return datasets


class PopulateSharedTorch(PopulateShared[ManagerT]):
    @property
    @abstractmethod
    def path(self) -> PathLike:
        raise NotImplementedError

    def shared_in(self, _) -> Mapping[str, TensorV]:
        return torch.load(self.path)
