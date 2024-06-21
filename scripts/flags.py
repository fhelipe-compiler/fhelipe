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

import argparse
from pathlib import Path
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--lattigo", action="store_true")
parser.add_argument("--clean", action="store_true")
parser.add_argument("--ds", type=int, default=0)
parser.add_argument(
    "--folder",
    type=str,
    default=utils.results_root("fhelipe_experiments"),
)
parser.add_argument("--program", type=str, default="")
parser.add_argument("--fhelipe", type=str, default=Path(__file__).parent.parent.resolve())
args = parser.parse_args()
ct_type = "lattigo" if args.lattigo else "clear"
ds = str(args.ds)
folder = Path(args.folder)
fhelipe_path = Path(args.fhelipe)
clean = args.clean
