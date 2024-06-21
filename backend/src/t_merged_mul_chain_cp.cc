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

#include "include/t_merged_mul_chain_cp.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <vector>

#include "include/ct_program.h"
#include "include/laid_out_tensor.h"
#include "include/t_op_utils.h"
#include "include/tensor_layout.h"

namespace fhelipe {

class CtOp;

namespace {

void CheckLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) {
  CHECK(input_layout.TotalChunks() == output_layout.TotalChunks());
}

}  // namespace

TOp::LaidOutTensorCt TMergedMulChainCP::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  CHECK(input_tensors[0].Layout() == input_layout_);
  return TOp::LaidOutTensorCt{Estd::transform(
      input_tensors[0].Chunks(), Estd::indices(output_layout_.TotalChunks()),
      [this](const auto& chunk, auto idx) {
        return LaidOutChunk{output_layout_, output_layout_.ChunkOffsets()[idx],
                            chunk.Chunk()};
      })};
}

void TMergedMulChainCP::SetLayouts(const TensorLayout& input_layout,
                                   const TensorLayout& output_layout) {
  input_layout_ = input_layout;
  output_layout_ = output_layout;
  CheckLayouts(input_layout_, output_layout_);
}

bool TMergedMulChainCP::EqualTo(const TOp& other) const {
  const auto* t_merged_mul_chain_cp =
      dynamic_cast<const TMergedMulChainCP*>(&other);
  return t_merged_mul_chain_cp &&
         t_merged_mul_chain_cp->OutputLayout() == OutputLayout();
}

TMergedMulChainCP::TMergedMulChainCP(const TensorLayout& input_layout,
                                     const TensorLayout& output_layout)
    : input_layout_(input_layout), output_layout_(output_layout) {
  CheckLayouts(input_layout_, output_layout_);
}

}  // namespace fhelipe
