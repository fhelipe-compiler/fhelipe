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

import math
from os import PathLike, fspath, path
from pathlib import Path
from typing import Dict, Final, Iterator, Mapping, TextIO, Union

import numpy as np
import torch

from ..core.utils import TensorV, encode_seq

Id = Union[str, int]


def read_tensor(f: TextIO) -> TensorV:
    tokens = f.read().split()

    dim_cnt = int(tokens[0])
    int_tokens = [int(x) for x in tokens[: dim_cnt + 2]]
    float_tokens = [float(x) for x in tokens[dim_cnt + 2 :]]

    shape = int_tokens[1 : dim_cnt + 1]
    element_cnt = int_tokens[dim_cnt + 1]

    if element_cnt != math.prod(shape):
        raise ValueError(
            f"Invalid tensor file! Tensor shaped like {shape}, "
            "but contains only {element_cnt} elements.",
            f,
        )

    return torch.tensor(float_tokens, dtype=torch.float64).reshape(shape)


def write_tensor(f: TextIO, t: TensorV) -> None:
    arr = np.array(t, dtype=float)
    print(*encode_seq(arr.shape), file=f)
    print(arr.size, file=f)
    arr.tofile(f, sep=" ")
    f.write("\n")


class DataPath(PathLike):
    def __init__(self, path: PathLike):
        self.__path: Final = Path(path)

    def __fspath__(self) -> str:
        return fspath(self.__path)

    def __str__(self) -> str:
        return str(self.__path)

    @property
    def name(self) -> str:
        return self.__path.name

    @property
    def parent(self) -> "DataPath":
        return DataPath(self.__path.parent)

    def is_dir(self) -> bool:
        return self.__path.is_dir()

    def is_file(self) -> bool:
        return self.__path.is_file()

    def exists(self) -> bool:
        return self.__path.exists()

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> "DataPath":
        self.__path.mkdir(parents=parents, exist_ok=exist_ok)
        return self

    def unlink(self, missing_ok: bool = True) -> None:
        self.__path.unlink(missing_ok=missing_ok)

    def __truediv__(self, key: Union[Id, PathLike]) -> "DataPath":
        if isinstance(key, int):
            key = str(key)

        return DataPath(path=self.__path / Path(key))

    def symlink_to(self, target: PathLike) -> "DataPath":
        target = Path(target).resolve()
        rel_target = Path(path.relpath(target, start=self.parent))

        self.parent.mkdir()
        try:
            self.__path.symlink_to(rel_target)
        except FileExistsError as err:
            if not self.__path.is_symlink():
                raise err
            elif self.__path.resolve() != target:
                raise err

        return self

    def open(self, *args, **kwargs) -> TextIO:
        self.parent.mkdir()
        return self.__path.open(*args, **kwargs)

    def __iter__(self) -> Iterator["DataPath"]:
        if self.is_dir():
            names = (p.name for p in self.__path.iterdir())
            return (self / name for name in sorted(names))
        else:
            return iter(())

    def write_text(self, text: str) -> None:
        with self.open(mode="w") as f:
            f.write(text)

    def read_text(self) -> str:
        with self.open() as f:
            return f.read()

    def write_tensor(self, t: TensorV) -> None:
        with self.open(mode="w") as f:
            write_tensor(f, t)

    def read_tensor(self) -> TensorV:
        with self.open() as f:
            return read_tensor(f)

    def export_tensors(self, tensors: Mapping[str, TensorV]) -> None:
        for k, v in tensors.items():
            (self / k).write_tensor(v)

    def import_tensors(self) -> Dict[str, TensorV]:
        return {p.name: p.read_tensor() for p in self}
