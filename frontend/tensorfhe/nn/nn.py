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

from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Final, Mapping, Optional, Union, overload

import torch

from ..core import Input, Shape, Tensor, TensorV, TorchInput, op
from .module import Preprocess, WeakModule, WrapperModule


class NN(WeakModule):
    @property
    @abstractmethod
    def input_shape(self) -> Shape:
        raise NotImplementedError

    def load(self, inputs: Mapping[str, TensorV]) -> None:
        missing, _ = self.torch.load_state_dict(
            OrderedDict(inputs), strict=False
        )
        if missing:
            raise ValueError("Missing weights", missing)

    def state(self) -> Dict[str, TensorV]:
        return self.torch.state_dict()


class SimpleNN(NN):
    def __init__(
        self,
        *,
        module: WrapperModule,
        input_shape: Shape,
        preprocess: torch.nn.Module = torch.nn.Identity(),
    ):
        """Construct a simple neural network

        Args:
            module: The neural network.
            preprocess: Additional layers applied to the input before encrypting
                and passing it to `module`.
            input_shape: Shape of the input (before preporcessing).
        """
        super().__init__()

        self.pre: Final = Preprocess(preprocess)
        self.nn: Final = module
        self.__input_shape: Final = input_shape

    @property
    def input_shape(self) -> Shape:
        return self.__input_shape

    @overload
    def forward(self, x: TorchInput) -> TorchInput:
        ...

    @overload
    def forward(self, x: Input) -> Tensor:
        ...

    def forward(self, x: Input) -> Tensor:
        x = self.pre(x)
        x_enc = x.enc()
        x_enc = self.nn(x_enc)
        return x_enc
