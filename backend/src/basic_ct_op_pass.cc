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

#include "include/basic_ct_op_pass.h"

#include <cwchar>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/ct_program.h"
#include "include/dag.h"
#include "include/debug_info.h"
#include "include/dictionary.h"
#include "include/extended_std.h"
#include "include/laid_out_tensor.h"
#include "include/program_context.h"
#include "include/t_op.h"
#include "include/utils.h"

namespace fhelipe {

BasicCtOpPass::BasicCtOpPass(const ProgramContext& context,
                             std::unique_ptr<Dictionary<ChunkIr>>&& chunk_dict)
    : context_(context), chunk_dict_(std::move(chunk_dict)) {}

CtOpPassOutput BasicCtOpPass::DoPass(const CtOpPassInput& in_dag) {
  ct_program::CtProgram ct_program(context_, *chunk_dict_);
  std::unordered_map<const Node<LeveledTOp>*, TOp::LaidOutTensorCt>
      old_to_new_nodes;
  std::unordered_map<const Node<CtOp>*, const Node<LeveledTOp>*>
      new_to_old_nodes;
  auto nodes = in_dag.NodesInTopologicalOrder();
  int count = 0;
  for (const auto& node : nodes) {
    LOG(INFO) << count++ << " / " << nodes.size();
    WriteStream(LOG(INFO), node->Value());
    auto parents = Estd::values_from_keys(old_to_new_nodes, node->Parents());
    TOp::LaidOutTensorCt new_tensor =
        node->Value().AmendCtProgram(ct_program, parents);
    for (const auto& chunk : new_tensor.Chunks()) {
      new_to_old_nodes.emplace(chunk.Chunk().get(), node.get());
    }
    old_to_new_nodes.emplace(node.get(), new_tensor);
  }

  // Assign ancestor ids
  for (const auto& node :
       ct_program.GetDag().NodesInReverseTopologicalOrder()) {
    if (Estd::contains_key(new_to_old_nodes, node.get())) {
      node->AddAncestor(new_to_old_nodes.at(node.get())->NodeId());
    } else {
      // Some nodes end up having all their children killed and not cleaned
      // up... there should be a pass to clean them up, but for now I just
      // don't register their ancestors.
      if (!node->Children().empty() &&
          !(*node->Children().begin())->Ancestors().empty()) {
        node->AddAncestor((*node->Children().begin())->Ancestors().at(0));
      }
    }
  }
  return ct_program;
}

}  // namespace fhelipe
