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

import os
import time
from pathlib import Path

from utils import cmd
import flags
import os
import random

shared_in_alias = {
    "fhelipe_resnet": ["resnet-weights"],
    "manual_resnet": ["resnet-weights"],
    "fhelipe_rnn": ["install-weights"],
    "manual_rnn": ["install-weights"],
}
ds_in_alias = {
    "fhelipe_resnet": ["test-in"],
    "manual_resnet": ["test-in"],
    "fhelipe_rnn": ["test-in"],
    "manual_rnn": ["test-in"],
    "fhelipe_logreg": ["init-log-reg"],
    "manual_logreg": ["init-log-reg"],
}


def ds_flag(benchmark_name, ds_number):
    return [] if "logreg" in benchmark_name else ["--ds", ds_number]


def requires_shared_setup(benchmark_name):
    benchmarks_with_shared_in = [
        "fhelipe_resnet",
        "fhelipe_rnn",
        "manual_resnet",
        "manual_rnn",
    ]
    return any(benchmark_name == x for x in benchmarks_with_shared_in)


def run_sample(compiled_program, dataset_number, ct_type):
    ds_path = setup_sample(compiled_program, dataset_number)
    runtime = run_program(compiled_program, ds_path, ct_type)
    return runtime


# Atomic test and set
def try_make_file(filepath):
    try:
        os.open(filepath, os.O_CREAT | os.O_EXCL)
        return True
    except FileExistsError:
        return False


def setup_sample(compiled_program, dataset_number):
    ds_path = cmd(
        [
            "python",
        ]
        + compiled_program.source_code()
        + [
            "--root",
            compiled_program.root(),
            "in-ds",
            "--ds",
            dataset_number,
        ]
    )
    ds_path = Path(ds_path.decode("utf-8").strip())

    cmd(
        [
            "python",
        ]
        + compiled_program.source_code()
        + [
            "--root",
            compiled_program.root(),
            "backend-in",
            "--ds",
            dataset_number,
        ]
    )
    cmd(
        ["python"]
        + compiled_program.source_code()
        + [
            "--root",
            compiled_program.root(),
            "out-tfhe",
            "--ds",
            dataset_number,
        ]
    )
    return ds_path


def run_program(compiled_program, dataset_path, ct_type):
    cmd(
        [
            compiled_program.fhelipe_path() / "backend/release/encrypt",
            "--ct_type",
            ct_type,
            "--exe_folder",
            dataset_path,
        ]
        + compiled_program.lattigo_parameters().flags()
    )

    start = time.time()
    cmd(
        [
            compiled_program.fhelipe_path() / "backend/release/run",
            "--ct_type",
            ct_type,
            "--exe_folder",
            dataset_path,
        ]
    )
    runtime = time.time() - start
    print("Runtime", runtime)

    cmd(
        [
            compiled_program.fhelipe_path() / "backend/release/decrypt",
            "--ct_type",
            ct_type,
            "--exe_folder",
            dataset_path,
        ]
    )
    cmd(
        [
            compiled_program.fhelipe_path() / "backend/release/cross_check",
            "--exe_folder",
            dataset_path,
            "--no_complain",
        ]
    )

    return runtime
