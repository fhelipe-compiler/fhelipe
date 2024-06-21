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

#include "include/bootstrapping_pass_utils.h"

#include "include/extended_std.h"
#include "include/level.h"
#include "include/level_info.h"
#include "include/leveled_t_op.h"
#include "include/node.h"
#include "include/scaled_t_op.h"
#include "include/t_bootstrap_c.h"
#include "include/t_input_c.h"
#include "include/t_op.h"
#include "include/t_rescale_c.h"

namespace fhelipe {

Level MinLevel(const std::vector<LevelInfo>& level_infos) {
  return Estd::min_element(Estd::transform(
      level_infos,
      [](const auto& level_info) { return level_info.Level().value(); }));
}

std::vector<LevelInfo> ExtractLevelInfos(
    const std::vector<std::shared_ptr<Node<LeveledTOp>>>& nodes) {
  std::vector<LevelInfo> result;
  result.reserve(nodes.size());
  for (const auto& node : nodes) {
    result.push_back(node->Value().GetLevelInfo());
  }
  return result;
}

LevelInfo NodeLevelInfo(const Node<ScaledTOp>& scaled_node,
                        const std::vector<LevelInfo>& parents_level_info,
                        Level max_usable_levels) {
  const TOp* old_node = &scaled_node.Value().GetTOp();
  if (const auto* t_input_c = dynamic_cast<const TInputC*>(old_node)) {
    return {max_usable_levels, t_input_c->GetLogScale()};
  }
  if (const auto* t_bootstrap_c = dynamic_cast<const TBootstrapC*>(old_node)) {
    return {t_bootstrap_c->GetUsableLevels(), scaled_node.Value().LogScale()};
  }
  if (const auto* t_rescale_c = dynamic_cast<const TRescaleC*>(old_node)) {
    return {parents_level_info.at(0).Level() - Level(1),
            scaled_node.Value().LogScale()};
  }
  if (const auto* t_bootstrap_c = dynamic_cast<const TBootstrapC*>(old_node)) {
    return {t_bootstrap_c->GetUsableLevels(), scaled_node.Value().LogScale()};
  }
  return {MinLevel(parents_level_info), scaled_node.Value().LogScale()};
}

}  // namespace fhelipe
