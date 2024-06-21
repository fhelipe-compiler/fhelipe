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

from math import ceil, floor, log

import numpy as np

n = 2**15
g_max = 16


def count_carry_patterns(a, n_s=1, n_e=2 * n):
    i = np.arange(n)
    i_s = i - i % n_s + (i + a) % n_s
    i_e = i - i % n_e + (i + a) % n_e

    c = i_s ^ a ^ i_e
    return len(np.unique(c))


def min_stages(a):
    n_s, n_e = 1, 1

    stages = 1
    sum_g, cur_g = 0, 0
    # = to handle wrap arounds
    while n_e <= n:
        g = count_carry_patterns(a, n_s, n_e * 2)
        # print(n_s, n_e, g)

        if g > g_max:
            n_s = n_e
            stages += 1
            sum_g += cur_g
            # print("stage")
        else:
            n_e *= 2
            cur_g = g

    sum_g += cur_g
    return stages, sum_g


n = 2**15
l = []
# print(min_stages(3879))
# print(min_stages(1017))

for a in range(1, n):
    g = count_carry_patterns(a)
    s, w = min_stages(a)
    w_opt = ceil(g ** (1 / s) * s)

    if w > 2 * w_opt:
        print(a, g, s, w, w_opt, w / w_opt)

    if a % 1024 == 0:
        print(a)


# wors case: (-1597, 10923)
