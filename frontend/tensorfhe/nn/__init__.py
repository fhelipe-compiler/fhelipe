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

"""Neural network framework for FHE.

This module enables building neural networks that support both training in
PyTorch and inference in FHE.

It mimics the interface of PyTorch: it provides a set of primitive layers that
can be composed into more complex models by subclassing `nn.Module` and
overriding `forward()` (just like subclassing `torch.nn.Module`). `nn.SimpleNN`
proides a thin wrapper around a module that also supports preprocessing data
unencrypted.

To implement new primitive layers, `.module` provides additional base classes
that can be subclassed; `nn.NN` provides a more flexible neural network
interface.
"""
from .approx_relu import ApproxRelu
from .avg_pool import AvgPool2d, GlobalAvgPool2d
from .batch_norm import BatchNorm2d
from .conv import Conv2d, ConvBn2d
from .dropout import Dropout
from .herpn import HerPN2d
from .linear import Linear
from .module import Module, ModuleV, Preprocess, WeakModule
from .nn import NN, SimpleNN
from .parameter import Parameter
from .sequential import Sequential
