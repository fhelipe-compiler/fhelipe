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

#include "include/layout_utils.h"

#include <unistd.h>

#include "include/t_input_c.h"
#include "include/t_layout_conversion_c.h"
#include "include/utils.h"

namespace fhelipe {

std::shared_ptr<Node<TOp>> AddLayoutConversion(
    Dag<TOp>& dag, const std::shared_ptr<Node<TOp>>& node,
    const TensorLayout& output_layout) {
  if (node->Value().OutputLayout() == output_layout) {
    return node;
  }
  return dag.AddNode(std::make_unique<TLayoutConversionC>(
                         node->Value().OutputLayout(), output_layout),
                     {node});
}

bool HasLinearChainToInput(std::shared_ptr<Node<TOp>> node) {
  while (node->Children().size() <= 1 &&
         Estd::vector_to_set(node->Parents()).size() == 1) {
    node = node->Parents().at(0);
  }
  return dynamic_cast<const TInputC*>(&node->Value());
}

std::vector<TensorLayout::LayoutBit> ChunkBits(
    std::vector<TensorLayout::LayoutBit> layout_bits, ChunkSize chunk_size) {
  layout_bits = Estd::reverse(layout_bits);
  while (chunk_size.value() > (1 << layout_bits.size())) {
    layout_bits.push_back(std::nullopt);
  }
  layout_bits = Estd::reverse(layout_bits);
  int top_idx = ceil_log2(chunk_size.value());
  return std::vector<TensorLayout::LayoutBit>(layout_bits.begin(),
                                              layout_bits.begin() + top_idx);
}

bool AllLayoutsMatch(const std::vector<std::shared_ptr<Node<TOp>>>& nodes) {
  return Estd::all_of(nodes, [&nodes](const auto& node) {
    return node->Value().OutputLayout() == nodes[0]->Value().OutputLayout();
  });
}

// Match layouts to first node, unless exactly one of the nodes has a linear
// chain up to an input; in that case, match layouts to the node that does
// not have the linear chain.
std::vector<std::shared_ptr<Node<TOp>>> MatchLayoutsForHoisting(
    Dag<TOp>& dag, const std::vector<std::shared_ptr<Node<TOp>>>& nodes) {
  if (AllLayoutsMatch(nodes)) {
    // Fast path
    return nodes;
  }

  CHECK(nodes.size() == 2);

  auto [match_to_me, layout_convert_me] =
      HasLinearChainToInput(nodes.at(1))
          ? std::make_pair(nodes.at(0), nodes.at(1))
          : std::make_pair(nodes.at(1), nodes.at(0));

  auto converted = AddLayoutConversion(dag, layout_convert_me,
                                       match_to_me->Value().OutputLayout());
  std::vector<std::shared_ptr<Node<TOp>>> result = {match_to_me, converted};
  CHECK(AllLayoutsMatch(result));
  return result;
}

}  // namespace fhelipe
