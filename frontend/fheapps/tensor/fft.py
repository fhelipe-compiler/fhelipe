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
from math import pi
from typing import Union

import numpy as np
import tensorfhe as tfhe
import tensorfhe.app.actions as act
from tensorfhe import Vector


class Complex:
    def __init__(self, r: Vector, i: Vector):
        self.r = r
        self.i = i

    def __mul__(self, o: Union["Complex", Vector, float]) -> "Complex":
        if isinstance(o, Complex):
            return Complex(
                self.r * o.r + (-1) * self.i * o.i,
                self.r * o.i + self.i * o.r,
            )
        else:
            return Complex(self.r * o, self.i * o)

    def __add__(self, o: Union["Complex", Vector, float]) -> "Complex":
        if isinstance(o, Complex):
            return Complex(self.r + o.r, self.i + o.i)
        else:
            return Complex(self.r + o, self.i + o)

    def rotate(self, k: int) -> "Complex":
        return Complex(self.r.rotate(k), self.i.rotate(k))


def fft_dif(x_real: Vector) -> Complex:
    (n,) = x_real.shape
    x = Complex(x_real, tfhe.as_input(np.zeros(n)))

    k = n
    while k > 1:
        r = n // k

        group_mask_left = np.concatenate((np.ones(k // 2), np.zeros(k // 2)))
        group_mask_right = np.concatenate((np.zeros(k // 2), np.ones(k // 2)))

        mask_left = tfhe.as_input(np.tile(group_mask_left, r))
        mask_right = tfhe.as_input(np.tile(group_mask_right, r))

        x_l = x * mask_left
        x_r = x * mask_right

        x_sum = x_l + x_r.rotate(-k // 2)
        x_sub = x_l.rotate(k // 2) + (x_r * -1)
        x = x_sum + x_sub

        angle_step = 2j * pi / k
        angles = np.arange(k // 2) * angle_step
        twiddles = np.concatenate((np.ones(k // 2), np.exp(angles)))

        twiddles = np.tile(twiddles, r)
        twiddles = Complex(
            tfhe.as_input(np.real(twiddles)),
            tfhe.as_input(np.imag(twiddles)),
        )

        x = x * twiddles
        k //= 2

    return x


class FFT(tfhe.App):
    @classmethod
    def add_instance_cli_args(cls, parser):
        parser.add_argument("+N", type=int, help="Vector size")

    def __init__(self, N=2**17, **kwargs) -> None:
        x = tfhe.tensor(tfhe.stdin, (N,))
        y = fft_dif(x)

        super().__init__(id=f"{N}", out={"real": y.r, "imaginary": y.i})


if __name__ == "__main__":
    FFT.main()
