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
from typing import Final, Sequence, Tuple, Union

import torch

__name_re: Final = re.compile(r"^[\w.-]+$", flags=re.ASCII)
stdin: Final = "input"
stdout: Final = "result"


Shape = Tuple[int, ...]
IndexElement = Union[int, slice, "ellipsis"]
Index = Union[IndexElement, Sequence[IndexElement]]
Attributes = Sequence[Union[int, str]]

TensorV = torch.Tensor


def is_a_valid_name(name: str) -> bool:
    return __name_re.fullmatch(name) is not None


def is_pow_of_2(x: int) -> bool:
    return x == 2 ** (x - 1).bit_length()


def is_a_permutation(seq: Sequence[int]) -> bool:
    return set(seq) == set(range(len(seq)))


def encode_seq(seq: Attributes) -> Attributes:
    return (len(seq), *seq)


def seq_str(seq: Attributes) -> str:
    return " ".join(map(str, seq))


def dim_selection(dim: int, sel: IndexElement) -> Index:
    if dim < 0:
        raise ValueError("dim must be non-negative")
    return (slice(None),) * dim + (sel,)


def shift_wrap_selection(dimension: int, shift: int) -> Index:
    if shift >= 0:
        shift_slice = slice(None, shift)
    else:
        shift_slice = slice(shift, None)

    return dim_selection(dimension, shift_slice)
