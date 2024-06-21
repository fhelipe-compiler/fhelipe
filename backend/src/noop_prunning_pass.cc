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

#include "include/noop_prunning_pass.h"

#include <glog/logging.h>

#include <memory>

#include "include/add_cc.h"
#include "include/add_cp.h"
#include "include/add_cs.h"
#include "include/input_c.h"
#include "include/mul_cc.h"
#include "include/mul_cp.h"
#include "include/mul_cs.h"
#include "include/node.h"
#include "include/output_c.h"
#include "include/rotate_c.h"

namespace fhelipe {

std::shared_ptr<Node<CtOp>> OtherParentOfChild(
    const std::shared_ptr<Node<CtOp>>& parent,
    const std::shared_ptr<Node<CtOp>>& child) {
  CHECK(Estd::contains(child->Parents(), parent));
  auto others = Estd::filter(
      child->Parents(), [&parent](const auto& ptr) { return ptr != parent; });
  CHECK(others.size() == 1);
  return others.at(0);
}

void PruneDescendants(const std::shared_ptr<Node<CtOp>>& zero_c) {
  auto children = zero_c->Children();
  do {
    auto children = zero_c->Children();
    for (const auto& child : children) {
      if (const auto* ct_op = dynamic_cast<const RotateC*>(&child->Value())) {
        RemoveNode(*child);
      } else if (const auto* ct_op =
                     dynamic_cast<const OutputC*>(&child->Value())) {
        continue;
      } else if (const auto* ct_op =
                     dynamic_cast<const MulCP*>(&child->Value())) {
        RemoveNode(*child);
      } else if (const auto* ct_op =
                     dynamic_cast<const MulCS*>(&child->Value())) {
        RemoveNode(*child);
      } else if (const auto* ct_op =
                     dynamic_cast<const MulCC*>(&child->Value())) {
        RemoveParentChildEdge(*OtherParentOfChild(zero_c, child), *child);
        RemoveNode(*child);
      } else if (const auto* ct_op =
                     dynamic_cast<const AddCP*>(&child->Value())) {
        continue;
      } else if (const auto* ct_op =
                     dynamic_cast<const AddCS*>(&child->Value())) {
        continue;
      } else if (const auto* ct_op =
                     dynamic_cast<const AddCC*>(&child->Value())) {
        RemoveParentChildEdge(*zero_c, *child);
      }
    }
  } while (children != zero_c->Children());
}

void DoNoopPrunningPass(ct_program::CtProgram& ct_program) {
  auto& dag = ct_program.GetDag();

  auto sentinel = dag.Sentinel();

  auto zeroes = Estd::filter(sentinel->Children(), [](auto child) {
    return dynamic_cast<const ZeroC*>(&child->Value());
  });

  for (const auto& zero : zeroes) {
    PruneDescendants(zero);
  }
}

}  // namespace fhelipe
