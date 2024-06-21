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

from argparse import ArgumentParser

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


class ConvBn(nn.Module):
    def __init__(self, in_c, c, stride=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, c, kernel_size=3, stride=stride, bias=False, padding=1
        )
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_c, c, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBn(in_c, c, stride=stride), nn.ReLU(), ConvBn(c, c)
        )
        self.relu = nn.ReLU()

        self.in_c = in_c
        self.c = c
        self.stride = stride

    def forward(self, x):
        res = self.residual(x)

        sc = x
        if self.stride != 1:
            sc = sc[:, :, :: self.stride, :: self.stride]
            sc = nn.functional.pad(sc, (0, 0, 0, 0, 0, self.c - self.in_c))

        return self.relu(sc + res)


class ResNet(nn.Module):
    def __init__(self, n, num_classes):
        super().__init__()
        self.init = nn.Sequential(ConvBn(3, 16), nn.ReLU())

        self.stages = nn.Sequential(
            self.__make_stage(n, 16, 16, 1),
            self.__make_stage(n, 16, 32, 2),
            self.__make_stage(n, 32, 64, 2),
        )

        self.pool = nn.Sequential(
            nn.AvgPool2d((8, 8)),
            nn.Flatten(start_dim=1),
        )
        self.linear = nn.Linear(64, num_classes)

    def __make_stage(self, n, in_c, c, stride):
        modules = []
        prev_c = in_c

        for i in range(n):
            modules.append(BasicBlock(prev_c, c, stride))
            prev_c = c
            stride = 1

        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.init(x)
        x = self.stages(x)
        x = self.pool(x)
        x = self.linear(x)

        return x


def get_cifar(train):
    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return CIFAR10(
        root=".",
        train=train,
        transform=transform,
        download=True,
    )


def init():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    global model
    n = (layers - 2) // 6
    model = ResNet(n, 10).to(device)

    global opt
    opt = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001
    )

    global sch

    def lr_lambda(epoch):
        if epoch < epoch_cnt // 2:
            return 1
        elif epoch < epoch_cnt * 3 // 4:
            return 0.1
        else:
            return 0.01

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    global test_data, train_data
    train_data = list(get_cifar(train=True))
    test_data = list(get_cifar(train=False))


def data_loader(data):
    dl = DataLoader(data, shuffle=True, batch_size=batch_size, pin_memory=True)

    for x, y in dl:
        yield x.to(device), y.to(device)


def prediction_err(y, target):
    prediction = torch.argmax(y, 1)
    return (prediction != target).sum().item()


def train_epoch():
    input_transform = nn.Sequential(
        RandomHorizontalFlip(), RandomCrop((32, 32), padding=4)
    )
    training_model = nn.Sequential(input_transform, model)
    model.train()

    loss = 0
    error = 0

    for x, target in data_loader(train_data):
        opt.zero_grad()

        y = training_model(x)

        batch_loss = loss_criterion(y, target)
        batch_loss.backward()
        opt.step()

        loss += batch_loss.item()
        error += prediction_err(y, target) / len(train_data)

    train_loss.append(loss)
    train_error.append(error)

    print(
        f"Training {len(train_loss)}: loss={loss:.3f}; err={error * 100:.2f}%"
    )
    sch.step()


def test_epoch():
    model.eval()
    error = 0

    for x, target in data_loader(test_data):
        y = model(x)
        error += prediction_err(y, target) / len(test_data)

    test_error.append(error)
    print(f"Test {len(test_error)}: err={error * 100:.2f}%")


def save_csv():
    record_types = {
        "train_loss": train_loss,
        "train_error": train_error,
        "test_error": test_error,
    }

    records = []
    for t, l in record_types.items():
        for e, v in enumerate(l):
            records.append(
                {"layers": layers, "type": t, "epoch": e, "value": v}
            )
    df = pd.DataFrame(records)
    df.to_csv(f"resnet-{layers}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layers", "-l", type=int, default=20)
    args = parser.parse_args()

    layers = args.layers
    epoch_cnt = 180
    batch_size = 128
    loss_criterion = torch.nn.CrossEntropyLoss()

    init()

    train_loss = []
    train_error = []
    test_error = []

    for e in range(epoch_cnt):
        train_epoch()
        test_epoch()

    save_csv()
