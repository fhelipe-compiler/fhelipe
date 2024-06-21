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

#include "include/layout_hoisting_pass.h"

#include <include/tensor_layout.h>

#include "include/dag.h"
#include "include/extended_std.h"
#include "include/fill_gaps_layout_pass.h"
#include "include/layout_utils.h"
#include "include/t_layout_conversion_c.h"

namespace fhelipe {

int TotalLayoutConversionTentaclesEstimate(const TensorLayout& input_layout,
                                           const TensorLayout& output_layout) {
  CHECK(input_layout.ChunkSize() == output_layout.ChunkSize());
  int bit_discrepancy_count = Estd::count_if(
      Estd::indices(LogChunkSize(input_layout.ChunkSize()).value()),
      [&input_layout, &output_layout](int idx) -> bool {
        return input_layout.Bits()[idx].has_value() &&
               input_layout.Bits()[idx] != output_layout.Bits()[idx];
      });
  return input_layout.TotalChunks() * (1 << bit_discrepancy_count);
}

TensorLayout InverseLayout(const TOp& node, const TensorLayout& output_layout) {
  return FillGapsLayoutPass::InverseStaticGetOutputLayout(node, output_layout);
}

bool NotMoreExpensiveAfterSwap(const std::shared_ptr<Node<TOp>>& node) {
  auto parent = node->Parents().at(0);
  auto new_parent_input_layout =
      InverseLayout(parent->Value(), node->Value().OutputLayout());
  auto new_parent_output_layout = node->Value().OutputLayout();
  auto new_node_input_layout = parent->Parents().at(0)->Value().OutputLayout();
  const auto& new_node_output_layout = new_parent_input_layout;
  int new_tentacles = TotalLayoutConversionTentaclesEstimate(
      new_node_input_layout, new_node_output_layout);
  int old_tentacles = TotalLayoutConversionTentaclesEstimate(
      dynamic_cast<const TLayoutConversionC*>(&node->Value())->InputLayout(),
      node->Value().OutputLayout());
  return new_tentacles <= old_tentacles;
}

bool IsHoistableConversion(const std::shared_ptr<Node<TOp>>& node) {
  return dynamic_cast<const TLayoutConversionC*>(&node->Value()) &&
         Estd::vector_to_set(node->Parents()).size() == 1 &&
         node->Parents()[0]->Children().size() == 1 &&
         Estd::vector_to_set(node->Parents()[0]->Parents()).size() == 1 &&
         !dynamic_cast<const TInputC*>(&node->Parents().at(0)->Value()) &&
         NotMoreExpensiveAfterSwap(node);
}

std::shared_ptr<Node<TOp>> SwapConversionWithParent(
    std::shared_ptr<Node<TOp>> node) {
  auto parent = node->Parents().at(0);
  auto new_parent_input_layout =
      InverseLayout(parent->Value(), node->Value().OutputLayout());
  auto new_parent_output_layout = node->Value().OutputLayout();
  auto new_node_input_layout = parent->Parents().at(0)->Value().OutputLayout();
  const auto& new_node_output_layout = new_parent_input_layout;

  node->Value().SetLayouts(new_node_input_layout, new_node_output_layout);
  parent->Value().SetLayouts(new_parent_input_layout, new_parent_output_layout);

  SwapParentAndChild(parent, node);

  return node;
}

void HoistConversion(std::shared_ptr<Node<TOp>> node) {
  while (IsHoistableConversion(node)) {
    node = SwapConversionWithParent(node);
  }
}

LayoutOptimizerOutput LayoutHoistingPass::DoPass(
    const LayoutOptimizerInput& in_dag) {
  auto out_dag = CloneFromAncestor(in_dag);
  for (const auto& node : out_dag.NodesInTopologicalOrder()) {
    HoistConversion(node);
  }
  return out_dag;
}

}  // namespace fhelipe
