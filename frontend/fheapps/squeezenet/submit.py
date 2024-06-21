#!/usr/bin/env python3

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

from pathlib import Path
from subprocess import run


def experiments_root() -> Path:
    return Path(".") / "experiments"


def submit(
    epoch_cnt: int = 180,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    scheduler: str = "linear",
):
    train_id = (
        f"ep_{epoch_cnt}_lr_{lr}_{scheduler}_m_{momentum}_wd_{weight_decay}"
    )
    root = experiments_root() / train_id

    llsub_args = [
        "LLsub",
        "run.sh",
        "-s",
        "20",
        "-g",
        "volta:1",
        "-o",
        Path(".") / "logs" / train_id,
    ]
    train_args = [
        str(root),
        "--epoch-cnt",
        epoch_cnt,
        "--lr",
        lr,
        "--momentum",
        momentum,
        "--weight-decay",
        weight_decay,
        "--scheduler",
        scheduler,
    ]
    args = [*llsub_args, "--", *train_args]
    str_args = list(map(str, args))

    run(str_args)


if __name__ == "__main__":
    for sch in ("linear", "cosine", "step"):
        for lr in (0.1, 0.03, 0.01):
            submit(
                epoch_cnt=180,
                lr=lr,
                scheduler=sch,
            )
