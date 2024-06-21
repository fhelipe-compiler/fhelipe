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

#include "include/waterline_rescale.h"

#include <memory>
#include <unordered_map>

#include "include/dag.h"
#include "include/dag_depth_info.h"
#include "include/extended_std.h"
#include "include/program_context.h"
#include "include/scaled_t_op.h"
#include "include/t_add_cc.h"
#include "include/t_add_cp.h"
#include "include/t_add_csi.h"
#include "include/t_bootstrap_c.h"
#include "include/t_input_c.h"
#include "include/t_mul_cc.h"
#include "include/t_mul_cp.h"
#include "include/t_mul_csi.h"
#include "include/t_op.h"
#include "include/t_rescale_c.h"
#include "include/waterline_rescale.h"

namespace fhelipe {

namespace {

std::shared_ptr<Node<ScaledTOp>> WaterlineRescale(
    Dag<ScaledTOp>& dag, const std::shared_ptr<Node<ScaledTOp>>& node,
    LogScale ct_log_scale) {
  auto rescale_node = node;
  for (LogScale curr_log_scale = node->Value().LogScale();
       curr_log_scale >= 2 * ct_log_scale;
       curr_log_scale = curr_log_scale - ct_log_scale) {
    rescale_node = dag.AddNode(
        std::make_unique<ScaledTOp>(
            std::make_unique<TRescaleC>(
                rescale_node->Value().GetTOp().OutputLayout(), ct_log_scale),
            curr_log_scale - ct_log_scale),
        {rescale_node});
  }
  return rescale_node;
}

LogScale NodeLogScale(
    const TOp* old_node,
    const std::vector<std::shared_ptr<Node<ScaledTOp>>>& parents,
    LogScale ct_log_scale) {
  const std::vector<LogScale>& parents_log_scales = Estd::transform(
      parents, [](const auto& node) { return node->Value().LogScale(); });

  if (const auto* t_input_c = dynamic_cast<const TInputC*>(old_node)) {
    return t_input_c->GetLogScale();
  }
  if (const auto* t_add_cc = dynamic_cast<const TAddCC*>(old_node)) {
    return Estd::max_element(parents_log_scales);
  }
  if (const auto* t_add_cp = dynamic_cast<const TAddCP*>(old_node)) {
    return std::max(parents_log_scales[0], t_add_cp->PtTensorLogScale());
  }
  if (const auto* t_add_csi = dynamic_cast<const TAddCSI*>(old_node)) {
    return std::max(parents_log_scales[0], t_add_csi->Scalar().GetLogScale());
  }
  if (const auto* t_mul_cc = dynamic_cast<const TMulCC*>(old_node)) {
    return Estd::sum(parents_log_scales);
  }

  // TODO(nsamar): BackendMaskDepth() should be multiplied by the log scale of
  // backend-generated masks, not ct_log_scale. This scale of
  // backend-generated masks could be stored as part of the context
  return parents_log_scales[0] + old_node->AddedLogScale() +
         old_node->BackendMaskDepth() * ct_log_scale;
}

std::pair<std::shared_ptr<Node<ScaledTOp>>, std::shared_ptr<Node<ScaledTOp>>>
BuildNewNode(Dag<ScaledTOp>& dag, const std::shared_ptr<Node<TOp>>& old_node,
             const std::vector<std::shared_ptr<Node<ScaledTOp>>>& parents,
             LogScale ct_log_scale) {
  LogScale log_scale = NodeLogScale(&old_node->Value(), parents, ct_log_scale);
  auto new_node = dag.AddNode(
      std::make_unique<ScaledTOp>(old_node->Value().CloneUniq(), log_scale),
      parents, {old_node->NodeId()});

  return std::make_pair(new_node,
                        WaterlineRescale(dag, new_node, ct_log_scale));
}

}  // namespace

RescalingPassOutput WaterlineRescale::DoPass(const RescalingPassInput& in_dag) {
  Dag<ScaledTOp> dag;
  std::unordered_map<const Node<TOp>*, std::shared_ptr<Node<ScaledTOp>>>
      old_to_new_nodes;

  for (const auto& old_node : in_dag.NodesInTopologicalOrder()) {
    const auto& parents =
        Estd::values_from_keys(old_to_new_nodes, old_node->Parents());
    const auto [match_node, new_node] =
        BuildNewNode(dag, old_node, parents, context_.LogScale());
    old_to_new_nodes.emplace(old_node.get(), new_node);
  }

  return dag;
}

}  // namespace fhelipe
