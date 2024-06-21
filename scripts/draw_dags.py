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

import flags
from benchmark import craterlake_schedule, read_compiled
from default_parameters import benchmark_dict
from utils import add_parent_directory, cmd, results_root
from pathlib import Path

benchmark = benchmark_dict[flags.args.program]
root_path = Path(flags.args.folder)
benchmark.root = root_path
compiled = read_compiled(benchmark)


def omit_pass_filename_index(pass_filename):
    return "_".join(pass_filename.split("_")[1:])


def should_omit_pass(pass_filename):
    filters = ["basic_parser"]
    return any([x in pass_filename for x in filters])

destination_passes = [
    x.name
    for x in compiled.exe_folder.iterdir()
    if (
        x.is_file() and x.name.startswith("00") and not should_omit_pass(x.name)
    )
]

print(destination_passes)
for destination_pass in destination_passes:
    compiled.draw_dag(
        add_parent_directory(
            results_root("dags") / (destination_pass + ".pdf"), compiled.name()
        ),
        "basic_parser",
        omit_pass_filename_index(destination_pass),
    )
