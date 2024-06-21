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
from typing import Final, Mapping, Optional, Tuple

import torch

from ...core import Shape, TensorV
from ..datapath import DataPath
from ..exe_manager import ExeManager
from .action import Action
from .populate import PopulateDs


class GenRandIn(PopulateDs):
    help = "Generate random inputs"

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        self.__seed: Final = seed

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        super(GenRandIn, cls).add_cli_args(parser)
        parser.add_argument(
            "--seed", help="The seed for generating random tensors", type=int
        )

    def __rng(self) -> torch.Generator:
        rng = torch.Generator()
        if self.__seed is not None:
            rng.manual_seed(self.__seed)
        return rng

    def gen_tensor(self, rng: torch.Generator, shape: Shape) -> TensorV:
        return torch.rand(shape, generator=rng) * 2 - 1

    def ds_in(self, exe: ExeManager) -> Tuple[Mapping[str, TensorV]]:
        all_inputs = exe.df.in_shapes
        shared_inputs = set(in_.name for in_ in exe.datasets.shared.in_)
        inputs = {k: v for k, v in all_inputs.items() if k not in shared_inputs}

        rng = self.__rng()
        return (
            {
                k: self.gen_tensor(rng, shape)
                for k, shape in sorted(inputs.items())
            },
        )
