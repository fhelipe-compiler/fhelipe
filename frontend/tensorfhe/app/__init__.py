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

"""CLI for FHE apps.

FHE apps should subclass App (or NnApp for ..nn neural networks). See `fheapps`
for example usage.

Running the CLI involves applying an Action to an instance of the App. To see
information about each action and its arguments, run with `--help`.

To support custom Actions, Apps can add an Action subclass to their `actions`
class variable.
"""

from . import actions
from .app import App, InstanceId, NnApp
from .datapath import DataPath, read_tensor, write_tensor
from .dataset import DataSet
from .exe_manager import ExeManager, NnExeManager
