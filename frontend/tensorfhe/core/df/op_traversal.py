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

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    Dict,
    Final,
    Iterator,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
)

from ..op import Op

VT = TypeVar("VT")
TravT = TypeVar("TravT", bound="OpTraversal")


class OpTraversal(Mapping[Op, VT]):
    def __init__(self) -> None:
        super().__init__()
        self.__values: Final[Dict[Op, VT]] = {}

    @abstractmethod
    def _evaluate(self, op: Op, parent_values: Sequence[VT]) -> VT:
        raise NotImplementedError

    def _consume(self, node: Op) -> None:
        pass

    def _cache(self, node: Op) -> bool:
        return True

    def __get(self, start_node: Op) -> VT:
        # Run DFS iteratively to prevent stack overwflows
        tasks = [(True, start_node)]
        results = []

        while tasks:
            is_start, node = tasks.pop()

            if node in self.__values:
                results.append(self.__values[node])

                self._consume(node)
                if not self._cache(node):
                    del self.__values[node]

                continue

            if is_start:
                tasks.append((False, node))

                for p in node.parents:
                    tasks.append((True, p))
            else:
                parent_values = []
                for _ in node.parents:
                    parent_values.append(results.pop())

                value = self._evaluate(node, parent_values)
                results.append(value)

                self._consume(node)
                if self._cache(node):
                    self.__values[node] = value

        assert len(results) == 1
        return results[0]

    def __getitem__(self, node: Op) -> VT:
        return self.__get(node)

    def __iter__(self) -> Iterator[Op]:
        return iter(self.__values)

    def __len__(self) -> int:
        return len(self.__values)

    def traverse(self: TravT, nodes: Sequence[Op]) -> TravT:
        for n in nodes:
            _ = self[n]
        return self

    def sorted_items(self) -> Sequence[Tuple[Op, VT]]:
        key_f = lambda kv: kv[1]
        return sorted(self.items(), key=key_f)


class OpMultiTraversal(OpTraversal[Sequence[VT]]):
    def unique_values(self) -> Sequence[VT]:
        # Use `dict` instead of `set` to ensure a deterministic iteartion order.
        value_dict = {x: None for v in self.values() for x in v}
        return list(value_dict)


class ConsumersTraversal(OpTraversal[List[Op]]):
    def _evaluate(self, op: Op, parents: Sequence[List[Op]]) -> List[Op]:
        for p in parents:
            p.append(op)
        return []


GcTravT = TypeVar("GcTravT", bound="GcOpTraversal")


class GcOpTraversal(OpTraversal[VT]):
    def __init__(self) -> None:
        super().__init__()

        self.__cached: Mapping[Op, None] = {}
        self.__consumers: Mapping[Op, List] = {}

    def _consume(self, node: Op) -> None:
        if l := self.__consumers.get(node, []):
            l.pop()

    def _cache(self, node: Op) -> bool:
        return (node in self.__cached) or bool(self.__consumers.get(node, []))

    def traverse(self: GcTravT, nodes: Sequence[Op]) -> GcTravT:
        self.__consumers = ConsumersTraversal().traverse(nodes)
        self.__cached = {n: None for n in nodes}

        return super().traverse(nodes)
