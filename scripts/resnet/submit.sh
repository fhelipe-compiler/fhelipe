#!/bin/bash

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

#SBATCH -c 4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=20,32,44,56,110

source /etc/profile
module add anaconda/2022b

LAYERS=$SLURM_ARRAY_TASK_ID
echo "Training Resnet-$LAYERS"
python resnet.py --layers $LAYERS
