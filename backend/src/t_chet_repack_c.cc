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

#include "include/t_chet_repack_c.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <vector>

#include "include/chet_layout_pass.h"
#include "include/chunk_size.h"
#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/laid_out_tensor.h"
#include "include/t_layout_conversion_c.h"
#include "include/tensor_layout.h"

namespace fhelipe {
class CtOp;

TensorLayout ChetRepackedLayout(const LogChunkSize& log_chunk_size,
                                const Shape& shape) {
  auto layout_bits = ChetLayoutPass::DefaultLayoutBits(shape);
  return {shape, ChetLayoutPass::ChetChunkBits(layout_bits,
                                               ChunkSize(log_chunk_size))};
}

TChetRepackC::TChetRepackC(const TensorLayout& layout)
    : input_layout_(layout),
      output_layout_(ChetRepackedLayout(LogChunkSize(layout.ChunkSize()),
                                        layout.GetShape())) {}

TOp::LaidOutTensorCt TChetRepackC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors[0].Layout() == input_layout_);
  if (input_layout_ == output_layout_) {
    return input_tensors[0];
  }
  return TLayoutConversionC(input_layout_, output_layout_)
      .AmendCtProgram(ct_program, input_tensors);
}

void TChetRepackC::SetLayouts(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) {
  CHECK(output_layout ==
        ChetRepackedLayout(LogChunkSize(input_layout.ChunkSize()),
                           input_layout.GetShape()));
  input_layout_ = input_layout;
  output_layout_ = output_layout;
}

bool TChetRepackC::EqualTo(const TOp& other) const {
  const auto* t_chet_repack_c = dynamic_cast<const TChetRepackC*>(&other);
  return t_chet_repack_c && t_chet_repack_c->InputLayout() == InputLayout();
}

}  // namespace fhelipe
