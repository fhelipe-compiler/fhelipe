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

from typing import FrozenSet, Type

from .action import Action, GenericAction, NnAction, RunResultT
from .base_actions import *
from .base_nn_actions import *
from .nn_data import *
from .output_transforms import *
from .populate import *
from .random_inputs import *
from .test import *
from .train import *

ActionT = Type[Action]
NnActionT = Type[NnAction]

base_actions: FrozenSet[ActionT] = frozenset(
    (GenDataflow, CountUsefulOps, InitDs, InitShared, BackendIn, GenOutTfhe)
)
common_actions: FrozenSet[ActionT] = frozenset(
    (*base_actions, GenRandIn, PopulateSharedNoOp)
)
common_nn_actions: FrozenSet[NnActionT] = frozenset(
    (*base_actions, GenOutTorch, GenRandWeights, ClearTraining, InstallWeights)
)
