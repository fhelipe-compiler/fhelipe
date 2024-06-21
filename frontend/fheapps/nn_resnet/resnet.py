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
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path

import tensorfhe as tfhe
import tensorfhe.app.actions as act
import torch
from tensorfhe import nn
from tensorfhe.nn import ModuleV
from torchvision.transforms import RandomCrop, RandomHorizontalFlip


class Shortcut(nn.Module):
    def __init__(self, c: int, stride: int):
        super().__init__()

        self.__stride = stride
        self.__out_c = c

    def forward(self, x: ModuleV) -> ModuleV:
        return (
            x.stride(dim=-1, by=self.__stride)
            .stride(dim=-2, by=self.__stride)
            .extend(dim=-3, size=self.__out_c)
        )


class ReluBasicBlock(nn.Module):
    def __init__(self, in_c: int, c: int, stride: int = 1):
        super().__init__()

        self.residual = nn.Sequential(
            nn.ConvBn2d(in_c, c, kernel_size=3, stride=stride),
            nn.ApproxRelu(alpha=13),
            nn.ConvBn2d(c, c, kernel_size=3),
        )
        self.shortcut = Shortcut(c, stride)
        self.relu = nn.ApproxRelu()

    def forward(self, x: ModuleV) -> ModuleV:
        sc = self.shortcut(x)
        res = self.residual(x)
        return self.relu(sc + res)


class HerPNBasicBlock(nn.Module):
    def __init__(self, in_c: int, c: int, stride: int = 1):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, c, kernel_size=3, stride=stride, bias=False),
            nn.HerPN2d(c=c),
            nn.Conv2d(c, c, kernel_size=3, bias=False),
        )
        self.shortcut = Shortcut(c, stride)
        self.herpn = nn.HerPN2d(c=c)

    def forward(self, x: ModuleV) -> ModuleV:
        sc = self.shortcut(x)
        res = self.residual(x)
        return self.herpn(sc + res)


class ResNet(nn.Module):
    def __init__(
        self,
        n: int,
        num_classes: int,
        basic_block_f,
    ):
        super().__init__()

        self.__basic_block = basic_block_f

        self.init = nn.Sequential(
            nn.ConvBn2d(3, 16, kernel_size=3), nn.ApproxRelu()
        )

        self.stages = nn.Sequential(
            self.__make_stage(n, 16, 16, 1),
            self.__make_stage(n, 16, 32, 2),
            self.__make_stage(n, 32, 64, 2),
        )

        self.pool = nn.GlobalAvgPool2d()
        self.linear = nn.Linear(64, num_classes)

    def __make_stage(
        self, n: int, in_c: int, c: int, stride: int
    ) -> nn.Sequential:
        modules = []
        prev_c = in_c

        for i in range(n):
            modules.append(self.__basic_block(prev_c, c, stride))
            prev_c = c
            stride = 1

        return nn.Sequential(*modules)

    def forward(self, x: ModuleV) -> ModuleV:
        x = self.init(x)
        x = self.stages(x)
        x = self.pool(x)
        x = self.linear(x)

        return x


class ResNetPreprocess(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Sequential(
            RandomHorizontalFlip(), RandomCrop((32, 32), padding=4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.module(x)
        else:
            return x


class ResNetTrain(act.ArgMaxOutputMixin, act.Cifar10Mixin, act.TrainNn):
    batch_size = 128
    momentum = 0.9

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        super(ResNetTrain, cls).add_cli_args(parser)

        g = parser.add_argument_group("Training Options")
        g.add_argument("--epoch-cnt", type=int)
        g.add_argument("--lr", type=float, help="Initial learning rate")
        g.add_argument("--weight-decay", type=float)

    def __init__(
        self,
        epoch_cnt: int = 180,
        lr: float = 0.1,
        weight_decay: float = 0.0001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch_cnt = epoch_cnt
        self.lr = lr
        self.weight_decay = weight_decay

    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def loss_f(self) -> act.LossT:
        return torch.nn.CrossEntropyLoss()

    def scheduler(self, optimizer: torch.optim.Optimizer) -> act.Scheduler:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.epoch_cnt,
        )


class ResNetError(act.ArgMaxOutputMixin, act.Cifar10TestError):
    pass


class ResNetWeights(act.PopulateSharedCheckpoint):
    def __init__(self, activation: str, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = Path(__file__).parent / "weights" / f"{activation}.pt"


class ResNetApp(tfhe.NnApp):
    actions = {
        *act.common_nn_actions,
        act.Cifar10TestIn,
        ResNetError,
        ResNetTrain,
        ResNetWeights,
    }
    input_shape = act.Cifar10TestIn.input_shape
    basic_blocks = {
        "relu": ReluBasicBlock,
        "herpn": HerPNBasicBlock,
    }

    @classmethod
    def add_instance_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "+l",
            "++layers",
            choices=(20, 32, 44, 56, 110),
            type=int,
            default=20,
            help="Number of ResNet layers",
        )
        parser.add_argument(
            "+a",
            "++activation",
            choices=tuple(cls.basic_blocks.keys()),
            default="relu",
            help="Activation function to use",
        )

    def __init__(self, layers: int, activation: str, **kwargs):
        n = (layers - 2) // 6
        block = self.basic_blocks[activation]

        super().__init__(
            id=(activation, layers),
            nn=nn.SimpleNN(
                module=ResNet(n, 10, block),
                preprocess=ResNetPreprocess(),
                input_shape=act.Cifar10TestIn.input_shape,
            ),
        )


if __name__ == "__main__":
    ResNetApp.main()
