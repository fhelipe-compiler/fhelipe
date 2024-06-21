/** $lic$
 * Copyright (C) 2023-2024 by Massachusetts Institute of Technology
 *
 * This file is part of the Fhelipe compiler.
 *
 * Fhelipe is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, version 3.
 *
 * Fhelipe is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>. 
 */

#include "include/value_numbering_pass.h"

#include "include/dag.h"
#include "include/extended_std.h"
#include "include/fill_gaps_layout_pass.h"
#include "include/layout_utils.h"
#include "include/node.h"
#include "include/t_layout_conversion_c.h"

namespace fhelipe {

namespace {

std::vector<std::shared_ptr<Node<TOp>>> OnlyNodesWithAllParentsFrom(
    const std::vector<std::shared_ptr<Node<TOp>>>& nodes,
    const std::vector<std::shared_ptr<Node<TOp>>>& allowed_parents) {
  return Estd::filter(nodes, [&allowed_parents](const auto& node) {
    return Estd::all_of(node->Parents(),
                        [&allowed_parents](const auto& parent) {
                          return Estd::contains(allowed_parents, parent);
                        });
  });
}

std::vector<std::shared_ptr<Node<TOp>>> NodesWithSameParents(
    const std::shared_ptr<Node<TOp>>& t_op) {
  CHECK(!t_op->Parents().empty());

  const auto& parents = t_op->Parents();
  std::set<std::shared_ptr<Node<TOp>>> result = parents.at(0)->Children();
  for (const auto& parent : parents) {
    result = Estd::set_intersection(result, parent->Children());
  }
  return OnlyNodesWithAllParentsFrom(
      Estd::set_to_vector(Estd::set_difference(result, {t_op})),
      t_op->Parents());
}

}  // namespace

LayoutOptimizerOutput ValueNumberingPass::DoPass(
    const LayoutOptimizerInput& in_dag) {
  auto out_dag = CloneFromAncestor(in_dag);
  for (const auto& node : out_dag.NodesInTopologicalOrder()) {
    if (node->Parents().empty()) {
      continue;
    }
    for (const auto& deduplication_candidate : NodesWithSameParents(node)) {
      if (node->Value() == deduplication_candidate->Value()) {
        InheritChildren(*node, deduplication_candidate);
        for (int ancestor : node->Ancestors()) {
          deduplication_candidate->AddAncestor(ancestor);
        }
        RemoveNodeWithoutReassaigningChildren(node);
        break;
      }
    }
  }
  return out_dag;
}

}  // namespace fhelipe
