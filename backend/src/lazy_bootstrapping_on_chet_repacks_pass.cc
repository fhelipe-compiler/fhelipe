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

#include "include/lazy_bootstrapping_on_chet_repacks_pass.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/bootstrapping_pass_utils.h"
#include "include/constants.h"
#include "include/dag.h"
#include "include/extended_std.h"
#include "include/level_info.h"
#include "include/level_info_utils.h"
#include "include/leveled_t_op.h"
#include "include/pass_utils.h"
#include "include/program_context.h"
#include "include/scaled_t_op.h"
#include "include/t_add_cc.h"
#include "include/t_add_cp.h"
#include "include/t_add_csi.h"
#include "include/t_bootstrap_c.h"
#include "include/t_chet_repack_c.h"
#include "include/t_input_c.h"
#include "include/t_mul_cc.h"
#include "include/t_mul_cp.h"
#include "include/t_mul_csi.h"
#include "include/t_op.h"
#include "include/t_rescale_c.h"
#include "include/utils.h"
#include "latticpp/ckks/ciphertext.h"

namespace fhelipe {

namespace {

std::shared_ptr<Node<LeveledTOp>> AddBootstrappingNode(
    Dag<LeveledTOp>& dag, const std::shared_ptr<Node<LeveledTOp>>& new_node,
    const LevelInfo& bootstrapped_level_info) {
  return dag.AddNode(std::make_unique<LeveledTOp>(
                         std::make_unique<TBootstrapC>(
                             new_node->Value().GetTOp().OutputLayout(),
                             bootstrapped_level_info.Level()),
                         bootstrapped_level_info),
                     {new_node});
}

bool RequiresLazyBootstrapping(const Node<ScaledTOp>& old_node) {
  // Bootstrap all TChetRepackC nodes that are not followed by a repack,
  // and all TRescaleC's that are children of TChetRepackC
  return (dynamic_cast<const TChetRepackC*>(&old_node.Value().GetTOp()) &&
          !old_node.Children().empty() &&
          !dynamic_cast<const TRescaleC*>(
              &(*old_node.Children().begin())->Value().GetTOp())) ||
         (dynamic_cast<const TRescaleC*>(&old_node.Value().GetTOp()) &&
          !old_node.Parents().empty() &&
          dynamic_cast<const TChetRepackC*>(
              &old_node.Parents()[0]->Value().GetTOp()));
}

std::shared_ptr<Node<LeveledTOp>> BootstrapIfNeeded(
    Dag<LeveledTOp>& dag, const std::shared_ptr<Node<ScaledTOp>>& old_node,
    const std::shared_ptr<Node<LeveledTOp>>& node,
    const LevelInfo& bootstrapped_level_info) {
  if (RequiresLazyBootstrapping(*old_node)) {
    return AddBootstrappingNode(dag, node, bootstrapped_level_info);
  }
  return node;
}

std::shared_ptr<Node<LeveledTOp>> BuildNewNode(
    Dag<LeveledTOp>& dag, const std::shared_ptr<Node<ScaledTOp>>& old_node,
    const std::vector<std::shared_ptr<Node<LeveledTOp>>>& parents,
    Level usable_levels, LogScale ct_log_scale) {
  auto new_node = dag.AddNode(
      std::make_unique<LeveledTOp>(
          old_node->Value().GetTOp().CloneUniq(),
          NodeLevelInfo(*old_node, ExtractLevelInfos(parents), usable_levels)),
      parents, {old_node->NodeId()});

  CHECK(new_node->Value().GetLevelInfo().Level() >= kMinLevel);

  return BootstrapIfNeeded(dag, old_node, new_node,
                           LevelInfo(usable_levels, ct_log_scale));
}

}  // namespace

LevelingPassOutput LazyBootstrappingOnChetRepacksPass::DoPass(
    const LevelingPassInput& in_dag) {
  std::unordered_map<const Node<ScaledTOp>*, std::shared_ptr<Node<LeveledTOp>>>
      old_to_new_nodes;
  Dag<LeveledTOp> out_dag;
  for (const auto& old_node : in_dag.NodesInTopologicalOrder()) {
    const auto& parents = ExtractParents(old_to_new_nodes, *old_node);
    // Ignore BootstrapC from frontend
    if (dynamic_cast<const TBootstrapC*>(&old_node->Value().GetTOp())) {
      old_to_new_nodes.emplace(
          old_node.get(), old_to_new_nodes.at(old_node->Parents().at(0).get()));
      continue;
    }
    const auto new_node =
        BuildNewNode(out_dag, old_node, parents, context_.UsableLevels(),
                     context_.LogScale());
    old_to_new_nodes.emplace(old_node.get(), new_node);
  }
  return out_dag;
}

}  // namespace fhelipe
