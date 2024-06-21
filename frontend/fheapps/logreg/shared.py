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

import csv
import logging
from os import PathLike
from pathlib import Path
from typing import Sequence, Tuple, TypeVar

import torch
from torch import Tensor

sigmoid_c = 0.5, -0.0843, 0, 0.0002
lr = 1.0
batch_size = 1024
batch_cnt = 11
feature_cnt = 197
it = 32


def load_csv(path: PathLike) -> Tuple[Tensor, Tensor]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        data = [[float(x) for x in row] for row in reader]
        t = torch.tensor(data, dtype=torch.float64)
        y = t[:, 0]
        X = t[:, 1:]

        return X, y


def normalize_data(X: Tensor, y: Tensor) -> Tensor:
    y = 2 * y - 1

    f_max, _ = X.abs().max(dim=0)
    div = torch.maximum(f_max, torch.tensor(1.0))
    X = X / div

    ones = torch.tensor(1).expand((X.size(0), 1))
    X = torch.cat((X, ones), dim=1)

    Z = X * y.unsqueeze(1)

    rng = torch.Generator().manual_seed(422)
    return Z[torch.randperm(Z.size(0), generator=rng)]


def analyze(pred: Tensor, Z: Tensor) -> Tuple[float, float]:
    m = Z @ pred
    accuracy = (m > 0).mean(dtype=torch.float64).item()
    loss = (1 + torch.exp(-m)).log().mean().item()

    return accuracy, loss


def load_data(rel_path: PathLike) -> Tensor:
    root = Path(__file__).parent
    X, y = load_csv(root / rel_path)
    return normalize_data(X, y)


def train_data() -> Tensor:
    return load_data(Path("data/MNIST_train.csv"))


def test_data() -> Tensor:
    return load_data(Path("data/MNIST_test.csv"))


def train_eta(it: int = it) -> Sequence[float]:
    eta = []
    alpha0, alpha1 = 0, 0
    for i in range(it):
        alpha0 = alpha1
        alpha1 = (1 + (1 + 4 * alpha0**2) ** 0.5) / 2
        eta.append((1 - alpha0) / alpha1)

    return eta


def batch(x: Tensor) -> Sequence[Tensor]:
    actual_b = len(x) // batch_size
    if actual_b != batch_cnt:
        logging.warning(f"Expected {batch_cnt} batches; got {actual_b}")

    trimmed = x[: batch_cnt * batch_size]
    return trimmed.chunk(batch_cnt)


BatchT = TypeVar("BatchT")


def train_batches(batches: Sequence[BatchT], it: int = it) -> Sequence[BatchT]:
    rng = torch.Generator().manual_seed(222)
    inds = torch.randint(0, len(batches), (it,), generator=rng).tolist()
    return [batches[i] for i in inds]
