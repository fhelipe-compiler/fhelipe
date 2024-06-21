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
from default_parameters import condor_fhelipe_squeezenet
from utils import cmd
from pathlib import Path
from benchmark import read_compiled

ds_path = Path(read_compiled(condor_fhelipe_squeezenet).exe_folder).parents[0] / "ds"

for sample in Path(ds_path).iterdir():
    if sample.name.startswith("0"):
        precision = cmd(["backend/release/cross_check", "--exe_folder", str(sample), "--verbose"])
        print(precision.decode("ascii"))
