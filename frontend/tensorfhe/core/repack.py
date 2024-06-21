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

from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from . import _vector
from .op import ChetRepack, Op

__manual_repack_depth: int = 0


def auto_repack_enabled() -> bool:
    assert __manual_repack_depth >= 0
    return __manual_repack_depth == 0


T = TypeVar("T")
P = ParamSpec("P")


def _manual_repack(f: Callable[P, T]) -> Callable[P, T]:
    def decorated(*args: P.args, **kwargs: P.kwargs) -> T:
        global __manual_repack_depth

        assert __manual_repack_depth >= 0
        __manual_repack_depth += 1

        result = f(*args, **kwargs)

        __manual_repack_depth -= 1
        assert __manual_repack_depth >= 0

        return result

    return decorated


VectorT = TypeVar("VectorT", bound="_vector.Vector")


def _result_repack(f: Callable[P, VectorT]) -> Callable[P, VectorT]:
    @_manual_repack
    def decorated(*args: P.args, **kwargs: P.kwargs) -> VectorT:
        return f(*args, **kwargs)._chet_repack()

    return decorated
