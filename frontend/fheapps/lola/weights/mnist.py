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

import re
from pathlib import Path
from urllib.request import urlopen

import torch

double_re = r"(-?\d*\.\d*(E[+-]\d+)?)"
comment_re = r"\s*(//.*\n)?\s*"


def load_source():
    url = "https://github.com/microsoft/CryptoNets/raw/92fae15162f2b339ce52592c485ed23558e44603/LowLatencyCryptoNets/Weights.cs"
    with urlopen(url) as f:
        return f.read().decode()


def get_array_str(s, array_name):
    array_prefix = r" \{ get; \} = new double\[\] \{\s*"
    numbers = double_re + r"(," + comment_re + double_re + r")*"
    array_end = comment_re + "\}"
    array_re = r"(?m)" + array_name + array_prefix + numbers + array_end

    result = re.search(array_re, s)
    return result.group()


def split_array_str(s):
    tokens = s.split(",")
    re_c = re.compile(double_re)
    return [float(re_c.search(t).group()) for t in tokens]


def get_array(array_name):
    array_s = get_array_str(source, array_name)
    return torch.tensor(split_array_str(array_s))


def weights_dict():
    d = {}
    conv_raw = get_array("Weights_0").reshape(5, 26)
    d["conv.b"] = conv_raw[:, -1]
    d["conv.w"] = conv_raw[:, :25].reshape((5, 1, 5, 5))

    d["fc1.w"] = (
        get_array("Weights_1")
        .reshape(5 * 13 * 13, 100)
        .T.reshape(100, 5, 13, 13)
    )
    d["fc1.b"] = get_array("Biases_2")

    d["fc2.w"] = get_array("Weights_3").reshape(10, 100)
    d["fc2.b"] = get_array("Biases_3")

    return d


if __name__ == "__main__":
    source = load_source()
    d = weights_dict()
    torch.save(d, Path(__file__).parent / "mnist.pt")
