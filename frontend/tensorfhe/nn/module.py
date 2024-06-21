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

from abc import ABC, abstractmethod
from typing import (
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch

from ..core import Input, Shape, Tensor, TorchInput, public_in


class WeakWrapperModule(ABC):
    def __init__(self, module: torch.nn.Module):
        # use `__setattr__` so that __torch isn't tracked as a submodule of
        # itself.
        self.__torch: torch.nn.Module
        object.__setattr__(self, "_WeakWrapperModule__torch", module)

        self.__parent: Optional[Tuple[WeakWrapperModule, str]] = None

    def __set_parent(self, module: "WeakWrapperModule", name: str) -> None:
        if self.__parent is not None:
            raise ValueError
        else:
            self.__parent = (module, name)

    def __setattr__(self, name: str, value) -> None:
        if isinstance(getattr(self, name, None), WeakWrapperModule):
            raise AttributeError("Cannot override Module attribute", name, self)

        if isinstance(value, WeakWrapperModule):
            value.__set_parent(self, name)
            self.torch.add_module(name, value.torch)

        super().__setattr__(name, value)

    def __parent_names(self) -> Sequence[str]:
        l: List[str] = []
        par = self.__parent

        while par is not None:
            l.append(par[1])
            par = par[0].__parent

        l.reverse()
        return l

    def submodule_name(self, name: Optional[str] = None) -> str:
        names: List[str] = []
        if name is not None:
            names.append(name)

        par = self.__parent
        while par is not None:
            names.append(par[1])
            par = par[0].__parent

        return ".".join(reversed(names))

    def weights(self, name: str, shape: Shape) -> Input:
        key = self.submodule_name(name)
        return public_in(key, shape)

    @abstractmethod
    def forward_fhe(self, x: Input) -> Tensor:
        raise NotImplementedError

    @property
    def torch(self) -> torch.nn.Module:
        return self.__torch

    @overload
    def __call__(self, x: TorchInput) -> TorchInput:
        ...

    @overload
    def __call__(self, x: Input) -> Tensor:
        ...

    def __call__(self, x: Union[Input, TorchInput]) -> Tensor:
        if isinstance(x, TorchInput):
            return TorchInput(self.torch(x.tensor))
        else:
            return self.forward_fhe(x)


class WeakSplitModule(WeakWrapperModule):
    def __init__(self) -> None:
        module = torch.nn.Module()
        module.forward = self.forward_torch
        super().__init__(module)

    def __setattr__(self, name: str, value):
        if isinstance(
            getattr(self, name, None), (torch.nn.Module, torch.nn.Parameter)
        ):
            raise AttributeError(
                "Cannot override torch.nn.Module/Parameter attribute",
                name,
                self,
            )

        super().__setattr__(name, value)

        if isinstance(value, torch.nn.Module):
            self.torch.add_module(name, value)
        if isinstance(value, torch.nn.Parameter):
            self.torch.register_parameter(name, value)

    @abstractmethod
    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class WeakModule(WeakSplitModule):
    @overload
    def forward(self, x: TorchInput) -> TorchInput:
        ...

    @overload
    def forward(self, x: Input) -> Tensor:
        ...

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(TorchInput(x)).tensor

    def forward_fhe(self, x: Input) -> Tensor:
        return self.forward(x)


ModuleV = TypeVar("ModuleV", Tensor, Input, TorchInput)
FheV = TypeVar("FheV", Tensor, Input)


class WrapperModule(WeakWrapperModule):
    @abstractmethod
    def forward_fhe(self, x: FheV) -> FheV:
        raise NotImplementedError

    def __call__(self, x: ModuleV) -> ModuleV:
        # Repeat body from WeakWrapperModule so that mypy can type check it.
        if isinstance(x, TorchInput):
            return TorchInput(self.torch(x.tensor))
        else:
            return self.forward_fhe(x)


class SplitModule(WrapperModule, WeakSplitModule):
    pass


class Module(WrapperModule, WeakModule):
    @abstractmethod
    def forward(self, x: ModuleV) -> ModuleV:
        raise NotImplementedError

    def forward_fhe(self, x: FheV) -> FheV:
        return self.forward(x)


PreprocessV = TypeVar("PreprocessV", Input, TorchInput)


class Preprocess(WeakWrapperModule):
    def __call__(self, x: PreprocessV) -> PreprocessV:
        # Repeat body from WeakWrapperModule so that mypy can type check it.
        if isinstance(x, TorchInput):
            return TorchInput(self.torch(x.tensor))
        else:
            return self.forward_fhe(x)

    def forward_fhe(self, x: Input) -> Input:
        return x.apply_torch(module=self.torch, name=self.submodule_name())
