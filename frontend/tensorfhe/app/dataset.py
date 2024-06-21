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
from typing import Dict, Final, Iterator

from ..core import TensorV
from .datapath import DataPath, Id


class DataSet(DataPath):
    def __init__(self, path: PathLike):
        super().__init__(path)

        self.in_: Final = self / "in"
        self.in_sh: Final = self / "in_shared"
        self.ct_in: Final = self / "ct_unenc"
        self.pt_in: Final = self / "pt"
        self.tdf: Final = self / "t.df"
        self.out_tfhe: Final = self / "out_tfhe"
        self.out_torch: Final = self / "out_torch"

    def import_inputs(self) -> Dict[str, TensorV]:
        shared = self.in_sh.import_tensors()
        own = self.in_.import_tensors()

        dup_keys = set(shared.keys()) & set(own.keys())
        if dup_keys:
            raise ValueError(
                "Tensors appear in both `in` and `in_shared`", self, dup_keys
            )

        shared.update(own)
        return shared


class DataSetMap:
    def __init__(self, ds_root: PathLike, shared_root: PathLike):
        self.__root: Final = DataPath(ds_root)
        self.shared: Final = DataSet(shared_root)

    def __getitem__(self, key: Id) -> DataSet:
        ds = DataSet(self.__root / key)

        ds.in_sh.symlink_to(self.shared.in_)
        for f in ("enc.cfg", "ch_ir", "t.df", "rt.df"):
            (ds / f).symlink_to(self.shared / f)

        return ds

    def __iter__(self) -> Iterator[DataSet]:
        return (self[p.name] for p in self.__root)

    def __len__(self) -> int:
        return len(list(self.__root))
