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

#include "include/noop_leveling_pass.h"

#include "include/bootstrapping_pass_utils.h"
#include "include/leveled_t_op.h"
#include "latticpp/ckks/ciphertext.h"

namespace fhelipe {

LevelingPassOutput NoopLevelingPass::DoPass(const LevelingPassInput& in_dag) {
  Dag<LeveledTOp> out_dag;
  std::unordered_map<const Node<ScaledTOp>*, std::shared_ptr<Node<LeveledTOp>>>
      old_to_new_nodes;
  for (const auto& old_node : in_dag.NodesInTopologicalOrder()) {
    auto parents = ExtractParents(old_to_new_nodes, *old_node);
    auto level_infos = ExtractLevelInfos(parents);
    auto new_node =
        out_dag.AddNode(std::make_unique<LeveledTOp>(
                            old_node->Value().GetTOp().CloneUniq(),
                            NodeLevelInfo(*old_node, level_infos,
                                          context_.UsableLevels().value())),
                        parents, {old_node->NodeId()});
    old_to_new_nodes.emplace(old_node.get(), new_node);
  }
  return out_dag;
}

}  // namespace fhelipe
