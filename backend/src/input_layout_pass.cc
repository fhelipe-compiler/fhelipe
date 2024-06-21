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

#include "include/input_layout_pass.h"

#include <memory>

#include "include/t_layout_conversion_c.h"

namespace fhelipe {

namespace {

bool AllChildrenLayoutConversions(const std::shared_ptr<Node<TOp>>& node) {
  return Estd::all_of(node->Children(), [](const auto& child) {
    return dynamic_cast<const TLayoutConversionC*>(&child->Value());
  });
}

TensorLayout PickNewInputLayout(const std::shared_ptr<Node<TOp>>& node) {
  return (*(node->Children().begin()))->Value().OutputLayout();
}

}  // namespace

LayoutOptimizerOutput InputLayoutPass::DoPass(
    const LayoutOptimizerInput& in_dag) {
  auto out_dag = CloneFromAncestor(in_dag);
  auto inputs = out_dag.Sentinel()->Children();
  for (const auto& input : inputs) {
    if (AllChildrenLayoutConversions(input)) {
      auto new_layout = PickNewInputLayout(input);
      input->Value().SetLayouts(new_layout, new_layout);
      auto children = Estd::set_to_vector(input->Children());
      for (const auto& child : children) {
        if (child->Value().OutputLayout() == new_layout) {
          RemoveNode(*child);
        } else {
          child->Value().SetLayouts(new_layout, child->Value().OutputLayout());
        }
      }
    }
  }
  return out_dag;
}
}  // namespace fhelipe
