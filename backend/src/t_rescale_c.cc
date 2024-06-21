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

#include "include/t_rescale_c.h"

#include <vector>

#include "include/ct_op.h"
#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/level_info.h"

namespace fhelipe {

TOp::LaidOutTensorCt TRescaleC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  LevelInfo old_level_info =
      input_tensors[0].Chunks().at(0).Chunk()->Value().GetLevelInfo();
  LevelInfo level_info(old_level_info.Level() - Level(1),
                       old_level_info.LogScale() - rescale_amount_);
  const auto result = Estd::transform(
      input_tensors[0].Chunks(), [&ct_program, &level_info](auto chunk) {
        return TOp::LaidOutChunk(
            chunk.Layout(), chunk.Offset(),
            ct_program::CreateRescaleC(ct_program, level_info, chunk.Chunk()));
      });

  return TOp::LaidOutTensorCt{result};
}

void TRescaleC::SetLayouts(const TensorLayout& input_layout,
                           const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
}

bool TRescaleC::EqualTo(const TOp& other) const {
  const auto* t_rescale_c = dynamic_cast<const TRescaleC*>(&other);
  return t_rescale_c && t_rescale_c->OutputLayout() == OutputLayout() &&
         t_rescale_c->RescaleAmount() == RescaleAmount();
}

}  // namespace fhelipe
