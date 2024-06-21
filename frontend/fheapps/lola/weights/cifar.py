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
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import torch


def load_csv(url):
    with urlopen(url) as f:
        csv_str = f.read().decode()

    with StringIO(csv_str, newline="") as f:
        r = csv.reader(f)
        return [torch.tensor(list(map(float, l))) for l in r]


def weights_dict(weights, biases):
    d = {}
    # d["conv1.w"] = weights[0].reshape((83, 3, 8, 8))
    d["conv1.w"] = weights[0].reshape((83, 3, 8, 8)).transpose(-1, -2)
    # d["conv1.w"] = weights[0].reshape((3, 83, 8, 8)).transpose(0, 1)

    d["conv1.b"] = biases[0]

    # d["conv2.w"] = weights[1].reshape((112, 83, 10, 10))
    d["conv2.w"] = weights[1].reshape((112, 83, 10, 10)).transpose(-1, -2)
    # d["conv2.w"] = weights[1].reshape((83, 122, 10, 10)).transpose(0, 1)

    d["conv2.b"] = biases[1]
    # d["fc.w"] = weights[2].reshape((10, 112, 7, 7))
    d["fc.w"] = weights[2].reshape((10, 112, 7, 7)).transpose(-1, -2)
    # d["fc.w"] = weights[2].reshape((112 * 7 * 7, 10)).T.reshape((10, 112, 7, 7))
    d["fc.bb"] = biases[2]

    return d


if __name__ == "__main__":
    url_prefix = (
        "https://github.com/microsoft/CryptoNets/raw/"
        "92fae15162f2b339ce52592c485ed23558e44603/"
        "CifarCryptoNet/"
    )
    weights = load_csv(url_prefix + "CifarWeight.csv")
    biases = load_csv(url_prefix + "CifarBias.csv")

    d = weights_dict(weights, biases)
    torch.save(d, Path(__file__).parent / "cifar.pt")
