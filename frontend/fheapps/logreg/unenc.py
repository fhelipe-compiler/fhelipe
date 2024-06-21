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

import fheapps.logreg.shared as sh
import torch


def real_sigmoid(x):
    return 1 / (1 + x.exp())


def sigmoid(x):
    out_of_range_cnt = (x < -16).sum() + (x > 16).sum()
    if out_of_range_cnt.item() > 0:
        logging.warning("Sigmoid out of range: {out_of_range_cnt} / {len(x)}")

    return sum(c * x**i for i, c in enumerate(sh.sigmoid_c))


def log_reg(Z, g=sigmoid, it=sh.it):
    n, m = Z.shape

    v = torch.zeros(m, dtype=torch.float64)
    w = torch.zeros(m, dtype=torch.float64)

    batch_arr = sh.train_batches(sh.batch(Z), it)
    eta_arr = sh.train_eta(it)

    for i, Z_b, eta in zip(range(it), batch_arr, eta_arr):
        gamma = sh.lr / len(Z_b)

        m = Z_b @ v
        s = g(m)
        grad = Z_b.T @ s

        w_n = v + gamma * grad
        v_n = (1 - eta) * w_n + eta * w
        v, w = v_n, w_n

        a, l = sh.analyze(v, Z)
        logging.info(f"train i={i:3}: acc={a:.3f}; L={l:.3f}")

    return v


if __name__ == "__main__":
    logging.basicConfig(
        format="{levelname}:{lineno} {message}", style="{", level=logging.INFO
    )

    Z = sh.train_data()
    beta = log_reg(Z, g=sigmoid, it=32)

    Z_test = sh.test_data()
    a, l = sh.analyze(beta, Z_test)
    print(f"Test acc: {a:.3f}; loss={l:.3f}")
