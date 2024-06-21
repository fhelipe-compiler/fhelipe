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

import subprocess
from pathlib import Path
from os.path import expanduser
import getpass

def results_root(subpath = ""):
    root = Path(expanduser("~")) / subpath
    root.mkdir(parents=True, exist_ok=True)
    return root

def cmd(command, cwd=None, timeout=None):
    command = [str(x) for x in command]
    print(" ".join(command))
    if timeout == None:
        result = subprocess.run(
            command,
            check=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
        )
    else:
        result = subprocess.run(
            command,
            check=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            timeout=timeout,
        )
    return result.stdout


def add_parent_directory(path: Path, prefix):
    cmd(["mkdir", "-p", path.parents[0] / prefix])
    return path.parents[0] / prefix / path.name
