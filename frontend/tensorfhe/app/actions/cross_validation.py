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
from functools import cached_property
from typing import Iterable, Sequence, Tuple

import torch

from .utils import NnData, NnSample


class CrossValidationMixin:
    cross_validation_fraction: float = 0.2
    cross_validation_seed: int = 0

    @cached_property
    def __data_split(self) -> Tuple[NnData, NnData]:
        all_data: NnData = super().train_data()  # type: ignore[misc]
        test_size = round(len(all_data) * self.cross_validation_fraction)
        train_size = len(all_data) - test_size

        logging.info(
            f"Splitting data into test,train: {train_size},{test_size}"
        )

        rng = torch.Generator().manual_seed(self.cross_validation_seed)

        split: Sequence[Iterable[NnSample]] = torch.utils.data.random_split(
            all_data,  # type: ignore[arg-type]
            [train_size, test_size],
            generator=rng,
        )  # type: ignore[assignment]

        return tuple(split[0]), tuple(split[1])

    def train_data(self) -> NnData:
        return self.__data_split[0]

    def test_data(self) -> NnData:
        return self.__data_split[1]
