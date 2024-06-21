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
from typing import List, Mapping, Sequence

import fheapps.logreg.shared as sh
import tensorfhe as tfhe
import tensorfhe.app.actions as act


class LogRegAccuracy(act.Action):
    name = "accuracy"
    help = "Determine log reg accuracy"

    def __init__(self, output_type: str = "out_tfhe", **kwargs):
        self.__out_t = output_type

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "output_type",
            choices=["out", "out_tfhe"],
            help="Selects the outputs for which accuracy is determined",
            nargs="?",
        )

    def run(self, exe: tfhe.app.ExeManager) -> List[str]:
        results: List[str] = []
        for ds in exe.datasets:
            logging.info(f"Computing accuracy of {ds.name}")
            beta = (ds / self.__out_t / tfhe.stdout).read_tensor()
            beta = beta[: sh.feature_cnt]
            a, l = sh.analyze(beta, sh.test_data())

            results.append(f"ds/{ds.name}: Acc={a:.3f}; Loss={l:.3f}")

        return results


class InitLogReg(act.PopulateDs):
    help = "Setup MNIST dataset"

    def ds_in(self, _) -> Sequence[Mapping[str, tfhe.TensorV]]:

        Z = sh.train_data()
        batches = sh.batch(Z)

        return ({str(i): b for i, b in enumerate(batches)},)
