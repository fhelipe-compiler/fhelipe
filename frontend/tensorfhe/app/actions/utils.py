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

from pathlib import Path
from typing import Final, Sequence, Tuple
from urllib.request import urlopen

from ...core import TensorV

NnSample = Tuple[TensorV, TensorV]
NnData = Sequence[NnSample]


class AvgStats:
    def __init__(self, value: int = 0, weight: int = 0):
        self.num: Final = value
        self.denum: Final = weight

    def __add__(self, other: "AvgStats") -> "AvgStats":
        return AvgStats(self.num + other.num, self.denum + other.denum)

    def __float__(self) -> float:
        if self.denum:
            return float(self.num / self.denum)
        else:
            return float("nan")


def prediction_error(y: TensorV, target: TensorV) -> AvgStats:
    if y.size() != target.size():
        raise ValueError(
            "Mismatching prediciton and target shapes", y.size(), target.size()
        )

    incorrect_cnt = int((y != target).sum().item())
    total = target.numel()
    return AvgStats(incorrect_cnt, total)


def download_root() -> Path:
    this = Path(__file__).resolve()
    # frontend/tensorfhe/.cache/
    cache = this.parent / ".cache"
    cache.mkdir(parents=True, exist_ok=True)

    return cache


def download(url, save_path) -> None:
    with open(save_path, mode="wb") as save_f:
        with urlopen(url) as url_f:
            while data := url_f.read(128):
                save_f.write(data)
