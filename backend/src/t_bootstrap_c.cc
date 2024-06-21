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

#include "include/t_bootstrap_c.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>

#include "include/constants.h"
#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/laid_out_tensor.h"

namespace fhelipe {
class CtOp;

TBootstrapC::TBootstrapC(const TensorLayout& layout, const Level& usable_levels,
                         std::optional<bool> is_shortcut)
    : layout_(layout),
      usable_levels_(usable_levels),
      is_shortcut_(is_shortcut) {}

TOp::LaidOutTensorCt TBootstrapC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  CHECK(input_tensors[0].Layout() == layout_);

  auto result =
      Estd::transform(input_tensors[0].Chunks(), [&](const auto& chunk) {
        return TOp::LaidOutChunk(
            chunk.Layout(), chunk.Offset(),
            ct_program::CreateBootstrapC(
                ct_program,
                {usable_levels_,
                 chunk.Chunk()->Value().GetLevelInfo().LogScale()},
                chunk.Chunk()));
      });
  return TOp::LaidOutTensorCt{result};
}

void TBootstrapC::SetLayouts(const TensorLayout& input_layout,
                             const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
}

bool TBootstrapC::EqualTo(const TOp& other) const {
  const auto* t_bootstrap_c = dynamic_cast<const TBootstrapC*>(&other);
  return t_bootstrap_c && t_bootstrap_c->OutputLayout() == OutputLayout() &&
         GetUsableLevels() == t_bootstrap_c->GetUsableLevels();
}

}  // namespace fhelipe
