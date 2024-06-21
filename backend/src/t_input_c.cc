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

#include "include/t_input_c.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>

#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/io_spec.h"
#include "include/laid_out_tensor.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"

namespace fhelipe {

TOp::LaidOutTensorCt TInputC::CreateInputTensor(
    ct_program::CtProgram& ct_program, const LevelInfo& level_info) const {
  const auto& chunk_offsets = layout_.ChunkOffsets();
  auto result = Estd::transform(chunk_offsets, [&ct_program, &level_info, this](
                                                   const TensorIndex& offset) {
    return TOp::LaidOutChunk{
        layout_, offset,
        ct_program::CreateInputC(ct_program, level_info,
                                 IoSpec(name_, offset.Flat()))};
  });
  return TOp::LaidOutTensorCt{result};
}

TOp::LaidOutTensorCt TInputC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  (void)ct_program;
  (void)input_tensors;
  LOG(FATAL)
      << "For TInputC Call CreateInputTensor() instead of AmendCtProgram()!";
}

void TInputC::SetLayouts(const TensorLayout& input_layout,
                         const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
}

bool TInputC::EqualTo(const TOp& other) const {
  const auto* t_input_c = dynamic_cast<const TInputC*>(&other);
  return t_input_c && Name() == t_input_c->Name() &&
         GetLogScale() == t_input_c->GetLogScale();
}

}  // namespace fhelipe
