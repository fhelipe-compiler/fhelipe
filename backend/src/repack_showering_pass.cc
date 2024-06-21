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

#include "include/repack_showering_pass.h"

#include "include/node.h"
#include "include/t_op_embrio.h"
#include "include/t_stride_c.h"

namespace fhelipe {

EmbrioOptimizerOutput RepackShoweringPass::DoPass(
    const EmbrioOptimizerOutput& in_dag) {
  auto out_dag = CloneFromAncestor(in_dag);

  auto nodes = out_dag.NodesInTopologicalOrder();
  for (const auto& node : nodes) {
    auto new_node = std::make_shared<Node<TOpEmbrio>>(
        std::make_unique<TChetRepackCEmbrio>(node->Value().OutputShape()));
    InheritChildren(*node, new_node);
    AddParentChildEdge(node, new_node);
  }
  return out_dag;
}

}  // namespace fhelipe
