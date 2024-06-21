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

#include "include/merge_stride_chain_pass.h"

#include <include/node.h>
#include <include/t_op_embrio.h>
#include <include/t_stride_c.h>

namespace fhelipe {

namespace {

bool ParentAndMeAreAStrideChain(Node<TOpEmbrio>& node) {
  if (node.Parents().size() != 1) {
    return false;
  }
  auto parent = node.Parents()[0];
  if (dynamic_cast<const TStrideCEmbrio*>(&node.Value()) &&
      dynamic_cast<const TStrideCEmbrio*>(&parent->Value())) {
    return true;
  }
  return false;
}

std::vector<Stride> MergeStrides(const std::vector<Stride>& lhs,
                                 const std::vector<Stride>& rhs) {
  return Estd::transform(lhs, rhs, [](auto lhs, auto rhs) {
    return Stride(lhs.value() * rhs.value());
  });
}

}  // namespace

EmbrioOptimizerOutput MergeStrideChainPass::DoPass(
    const EmbrioOptimizerOutput& in_dag) {
  auto out_dag = CloneFromAncestor(in_dag);

  for (auto node : out_dag.NodesInTopologicalOrder()) {
    if (ParentAndMeAreAStrideChain(*node)) {
      auto parent = node->Parents()[0];
      AddNodeOnParentChildEdge(
          parent, node,
          std::make_shared<Node<TOpEmbrio>>(
              std::make_unique<TMergedStrideCEmbrio>(
                  parent->Value().InputShape(),
                  MergeStrides(
                      dynamic_cast<const TStrideCEmbrio*>(&node->Value())
                          ->Strides(),
                      dynamic_cast<const TStrideCEmbrio*>(&parent->Value())
                          ->Strides()))));
      RemoveNode(*parent);
      RemoveNode(*node);
    }
  }
  return out_dag;
}

}  // namespace fhelipe
