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
from math import pi, sqrt
from pathlib import Path
from typing import Final

import tensorfhe as tfhe
import tensorfhe.app.actions as act
import torch
from tensorfhe import TorchInput, lib, nn
from tensorfhe.nn import ModuleV
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR
from torchvision.transforms import RandomCrop, RandomHorizontalFlip


class Squeeze(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_c, out_channels=out_c, kernel_size=1, bias=False
        )
        self.hpn = nn.HerPN2d(c=out_c)

    def forward(self, x: ModuleV) -> ModuleV:
        x = self.conv(x)
        x = self.hpn(x)
        return x


class Expand(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c // 2,
            kernel_size=1,
            bias=False,
            stride=stride,
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c // 2,
            kernel_size=3,
            bias=False,
            stride=stride,
        )
        self.hpn = nn.HerPN2d(c=out_c)

    def forward(self, x: ModuleV) -> ModuleV:
        y1 = self.conv1(x)
        y2 = self.conv3(x)

        y = lib.concat(y1, y2, dim=-3)
        y = self.hpn(y)

        return y


class Fire(nn.Module):
    def __init__(self, in_c: int, s_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.squeeze = Squeeze(in_c=in_c, out_c=s_c)
        self.expand = Expand(in_c=s_c, out_c=out_c, stride=stride)

    def forward(self, x: ModuleV) -> ModuleV:
        x = self.squeeze(x)
        x = self.expand(x)
        return x


class SqueezeNet(nn.Sequential):
    def __init__(self, num_classes: int):
        super().__init__(
            nn.ConvBn2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=2
            ),
            Fire(64, 16, 128),
            Fire(128, 16, 128),
            # nn.AvgPool2d(kernel_size=2, stride=2),
            Fire(128, 32, 256, stride=2),
            Fire(256, 32, 256),
            nn.ConvBn2d(
                in_channels=256, out_channels=num_classes, kernel_size=1
            ),
            nn.GlobalAvgPool2d(),
        )


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


class SqueezeNetTrain(act.ArgMaxOutputMixin, act.Cifar10Mixin, act.TrainNn):
    batch_size = 128

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        super(SqueezeNetTrain, cls).add_cli_args(parser)

        g = parser.add_argument_group("Training Options")
        g.add_argument("--epoch-cnt", type=int)
        g.add_argument("--lr", type=float, help="Initial learning rate")
        g.add_argument("--momentum", type=float)
        g.add_argument("--weight-decay", type=float)
        g.add_argument("--scheduler", choices=cls.schedulers.keys())

    def __init__(
        self,
        epoch_cnt: int = 180,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        scheduler: str = "linear",
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.epoch_cnt = epoch_cnt
        self.lr: Final = lr
        self.momentum: Final = momentum
        self.weight_decay: Final = weight_decay

        if scheduler not in self.schedulers.keys():
            raise ValueError(
                "Unexpected scheduler", scheduler, list(self.schedulers.keys())
            )
        self.scheduler_type: Final = scheduler

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
        scheduler_f = self.schedulers[self.scheduler_type]
        return scheduler_f(self, optimizer)

    def cosine_scheduler(self, optimizer: Optimizer) -> act.Scheduler:
        return CosineAnnealingLR(optimizer, T_max=self.epoch_cnt)

    def linear_scheduler(self, optimizer: Optimizer) -> act.Scheduler:
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.epoch_cnt,
        )

    def step_scheduler(self, optimizer: Optimizer) -> act.Scheduler:
        def lr_lambda(epoch):
            if epoch < self.epoch_cnt // 2:
                return 1
            elif epoch < self.epoch_cnt * 3 // 4:
                return 0.1
            else:
                return 0.01

        return LambdaLR(optimizer, lr_lambda)

    schedulers: Final = {
        "linear": linear_scheduler,
        "cosine": cosine_scheduler,
        "step": step_scheduler,
    }

    def run(self, exe: tfhe.app.NnExeManager) -> float:
        logging.info(f"Training parameters:")
        logging.info(f"  sch={self.scheduler_type}")
        logging.info(f"  epoch_cnt={self.epoch_cnt}")
        logging.info(f"  lr={self.lr}")
        logging.info(f"  momentum={self.momentum}")
        logging.info(f"  w_decay={self.weight_decay}")

        return super().run(exe)


class SqueezeNetError(act.ArgMaxOutputMixin, act.Cifar10TestError):
    pass


class SqueezeNetWeights(act.PopulateSharedCheckpoint):
    checkpoint = Path(__file__).parent / "checkpoint.pt"


class SqueezeNetApp(tfhe.NnApp):
    actions = {
        *act.common_nn_actions,
        act.Cifar10TestIn,
        SqueezeNetError,
        SqueezeNetTrain,
        SqueezeNetWeights,
    }
    input_shape = act.Cifar10TestIn.input_shape

    def __init__(self, **kwargs):
        super().__init__(
            nn=nn.SimpleNN(
                module=SqueezeNet(10),
                preprocess=ResNetPreprocess(),
                input_shape=act.Cifar10TestIn.input_shape,
            ),
        )


if __name__ == "__main__":
    SqueezeNetApp.main()
