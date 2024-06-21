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

#include "include/t_output_c.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>

#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/io_spec.h"
#include "include/laid_out_tensor.h"
#include "include/tensor_index.h"

namespace fhelipe {

namespace {

LevelInfo GetLevelInfo(const TOp::LaidOutTensorCt& lot) {
  return lot.Chunks().at(0).Chunk()->Value().GetLevelInfo();
}

}  // namespace

TOp::LaidOutTensorCt TOutputC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  LevelInfo level_info = GetLevelInfo(input_tensors[0]);
  // TODO(nsamar): Offsets() vector argument to Estd::transform si redundant. So
  // remove. Check if other TOps have the same issue.
  auto result = Estd::transform(
      input_tensors[0].Chunks(), input_tensors[0].Offsets(),
      [&ct_program, &level_info, this](const TOp::LaidOutChunk& parent,
                                       const TensorIndex& offset) {
        const auto& ct_op = ct_program::CreateOutputC(
            ct_program, level_info, IoSpec(name_, offset.Flat()),
            parent.Chunk());
        return TOp::LaidOutChunk{parent.Layout(), parent.Offset(), ct_op};
      });
  return TOp::LaidOutTensorCt{result};
}

void TOutputC::SetLayouts(const TensorLayout& input_layout,
                          const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
}

bool TOutputC::EqualTo(const TOp& other) const {
  const auto* t_output_c = dynamic_cast<const TOutputC*>(&other);
  return t_output_c && t_output_c->OutputLayout() == OutputLayout() &&
         Name() == t_output_c->Name();
}

}  // namespace fhelipe
