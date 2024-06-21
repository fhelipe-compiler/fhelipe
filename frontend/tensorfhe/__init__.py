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

"""Fhelipe DSL library.

Submodules:
    - core: Types for specifing FHE programs.
    - lib: Functions that work on FHE datatypes.
    - nn: For building FHE neural networks.
    - app: CLI framework for FHE apps.

Submodules provide some additional documentation. For example usage, see
`fheapps`.
"""
from . import app, lib, nn
from .app import App, NnApp
from .core import *
