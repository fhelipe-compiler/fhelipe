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

#include "include/generic_layout_pass.h"

#include "chet_layout_pass.h"
#include "include/pass_utils.h"
#include "include/t_layout_conversion_c.h"

namespace fhelipe {

TensorLayout GenericLayoutPass::NodeLayout(
    const TOpEmbrio& old_node,
    const std::vector<std::shared_ptr<Node<TOp>>>& parents,
    ChunkSize chunk_size) const {
  if (dynamic_cast<const TInputCEmbrio*>(&old_node)) {
    return DefaultLayout(old_node.OutputShape(), chunk_size);
  }
  return parents[0]->Value().OutputLayout();
}

std::shared_ptr<Node<TOp>> GenericLayoutPass::BuildNewNode(
    Dag<TOp>& dag, const std::shared_ptr<Node<TOpEmbrio>>& old_node,
    std::vector<std::shared_ptr<Node<TOp>>> parents, ChunkSize chunk_size) {
  parents = MatchLayouts(dag, parents);
  TensorLayout input_layout =
      NodeLayout(old_node->Value(), parents, chunk_size);
  auto output_layout = GetOutputLayout(old_node, input_layout);
  return dag.AddNode(old_node->Value().GetTOp(input_layout, output_layout),
                     parents, {old_node->NodeId()});
}

LayoutPassOutput GenericLayoutPass::DoPass(const Dag<TOpEmbrio>& in_dag) {
  std::unordered_map<const Node<TOpEmbrio>*, std::shared_ptr<Node<TOp>>>
      old_to_new_nodes;
  Dag<TOp> out_dag;
  auto chunk_size = ChunkSize(Context().GetLogChunkSize());

  if (ignore_chet_repack_) {
    for (auto& old_node : in_dag.NodesInTopologicalOrder()) {
      if (dynamic_cast<const TChetRepackCEmbrio*>(&old_node->Value())) {
        RemoveNode(*old_node);
      }
    }
  }

  for (const auto& old_node : in_dag.NodesInTopologicalOrder()) {
    const auto parents = ExtractParents(old_to_new_nodes, *old_node);
    auto new_node = BuildNewNode(out_dag, old_node, parents, chunk_size);
    old_to_new_nodes.emplace(old_node.get(), new_node);
  }

  // Turn ChetRepack into a layout conversion so that it can be decomposed later
  if (ChetLayoutPass::RowMajorHack()) {
    for (auto& node : out_dag.NodesInTopologicalOrder()) {
      if (const auto* chet_repack =
              dynamic_cast<const TChetRepackC*>(&node->Value())) {
        if (chet_repack->InputLayout() != chet_repack->OutputLayout()) {
          node->SetValue(std::make_unique<TLayoutConversionC>(
              chet_repack->InputLayout(), chet_repack->OutputLayout()));
        } else {
          RemoveNode(*node);
        }
      }
    }
  }
  return out_dag;
}

}  // namespace fhelipe
