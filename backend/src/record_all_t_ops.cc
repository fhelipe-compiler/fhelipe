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

#include "include/record_all_t_ops.h"

#include <memory>
#include <string>
#include <vector>

#include "include/dag.h"
#include "include/t_op_embrio.h"

namespace fhelipe {

void RecordAllTOps(Dag<TOpEmbrio>& in_dag) {
  int idx = 0;
  for (const auto& node : in_dag.NodesInTopologicalOrder()) {
    if (!dynamic_cast<const TOutputCEmbrio*>(&node->Value())) {
      in_dag.AddNode(
          std::make_unique<TOutputCEmbrio>(TOutputCEmbrio(
              node->Value().OutputShape(), "__debug" + std::to_string(++idx))),
          {node});
    }
  }
}

}  // namespace fhelipe
