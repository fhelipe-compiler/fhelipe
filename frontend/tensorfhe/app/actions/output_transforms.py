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

from typing import Optional, Protocol

from ...core import TensorV


class HasNumClassesProtocol:
    num_classes: Optional[int] = None


class ArgMaxOutputMixin(HasNumClassesProtocol):
    def output_transform(self, x: TensorV) -> TensorV:
        x = x[..., : self.num_classes]
        return x.argmax(dim=-1)


class BinaryOutputMixin:
    def output_transform(self, x: TensorV) -> TensorV:
        return x > 0
