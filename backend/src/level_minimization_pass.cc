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

#include "include/level_minimization_pass.h"

#include "include/bootstrap_c.h"
#include "include/ct_op.h"
#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/input_c.h"
#include "include/output_c.h"
#include "include/pass_utils.h"
#include "include/rescale_c.h"

namespace fhelipe {

namespace {
Level GetMinLevel(const Node<CtOp>& node) {
  if (dynamic_cast<const OutputC*>(&node.Value())) {
    return Level(1);
  }
  auto children = Estd::set_to_vector(node.Children());
  children = Estd::filter(children, [](const auto& child) {
    return !dynamic_cast<const BootstrapC*>(&child->Value());
  });
  if (children.empty()) {
    return Level(1);
  }
  return Estd::max_element(Estd::transform(children, [](const auto& child) {
    auto child_level = child->Value().GetLevel();
    return dynamic_cast<const RescaleC*>(&child->Value())
               ? Level(child_level.value() + 1)
               : child_level;
  }));
}

}  // namespace

CtOpOptimizerOutput LevelMinimizationPass::DoPass(
    const CtOpOptimizerInput& in_dag) {
  auto out_dag = CloneFromAncestor(in_dag.GetDag());

  for (const auto& node : out_dag.NodesInReverseTopologicalOrder()) {
    auto new_level = GetMinLevel(*node);
    node->Value().SetLevelInfo({new_level, node->Value().LogScale()});
  }
  return ct_program::CtProgram{in_dag.GetProgramContext(),
                               in_dag.ChunkDictionary()->CloneUniq(),
                               std::move(out_dag)};
}

}  // namespace fhelipe
