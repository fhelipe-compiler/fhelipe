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

#include "include/t_replicate_dim_c.h"

#include <glog/logging.h>

#include "include/dimension_bit.h"
#include "include/extended_std.h"
#include "include/laid_out_tensor.h"
#include "include/raw_shift_acc.h"
#include "include/raw_shift_bit.h"
#include "include/shape.h"
#include "include/t_resize_dim_c.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_utils.h"
#include "include/utils.h"

namespace fhelipe {
class CtOp;
namespace ct_program {
class CtProgram;
}  // namespace ct_program

Shape GetOutputShapeTReplicateDimsC(Shape shape, int dimension, int multiple) {
  shape[dimension] = shape[dimension] * multiple;
  return shape;
}

void SanityCheckTReplicateDimCLayouts(const TensorLayout& input_layout,
                                      const TensorLayout& output_layout,
                                      int dimension, int multiple) {
  CHECK(dimension >= 0 && dimension < input_layout.GetShape().DimensionCount());
  CHECK(multiple >= 1);
  CHECK(input_layout.GetShape()[dimension] == 1);
  CHECK(output_layout.GetShape()[dimension] == multiple);
  // Check all dimensions except `dimension` have equal shape between
  // input_layout and output_layout
  for (auto idx : Estd::indices(input_layout.GetShape().DimensionCount())) {
    if (idx != dimension) {
      CHECK(input_layout.GetShape()[idx] == output_layout.GetShape()[idx]);
    }
  }
  CHECK(input_layout.GetShape().DimensionCount() ==
        output_layout.GetShape().DimensionCount());
}

void TReplicateDimC::SetLayouts(const TensorLayout& input_layout,
                                const TensorLayout& output_layout) {
  SanityCheckTReplicateDimCLayouts(input_layout, output_layout, dimension_,
                                   multiple_);
  input_layout_ = input_layout;
  output_layout_ = output_layout;
}

TReplicateDimC::TReplicateDimC(const TensorLayout& input_layout,
                               const TensorLayout& output_layout, int dimension,
                               int multiple)
    : input_layout_(input_layout),
      output_layout_(output_layout),
      dimension_(dimension),
      multiple_(multiple) {
  SanityCheckTReplicateDimCLayouts(input_layout, output_layout, dimension,
                                   multiple);
}

bool TReplicateDimC::CanSkipResize() const {
  for (const auto& chunk : input_layout_.ChunkOffsets()) {
    int idx = Estd::find_index_pred(
        output_layout_.ChunkOffsets(), [&chunk](const auto& out_chunk) {
          return out_chunk.DimensionIndices() == chunk.DimensionIndices();
        });
    if (idx == output_layout_.ChunkOffsets().size()) {
      return false;
    }
  }
  return true;
}

TOp::LaidOutTensorCt TReplicateDimC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  const auto& input_tensor = input_tensors[0];
  CHECK(input_tensor.Layout() == input_layout_);
  auto output_layout = OutputLayout();

  // TODO(nsamar): Refactor. The purpose of the if is to shave off a level by
  // doing a poor-man's version of resize that avoids masking. This poor-man's
  // resize can only be applied when CanSkipResize() is true.
  auto result = input_tensor;
  if (CanSkipResize()) {
    std::vector<TOp::LaidOutChunk> sum =
        ZeroLaidOutTensor(input_tensor.Chunks().at(0).Chunk(), output_layout);
    for (const auto& chunk : input_tensor.Chunks()) {
      int sum_idx = Estd::find_index_pred(sum, [&chunk](const auto& sum_chunk) {
        return sum_chunk.Offset().DimensionIndices() ==
               chunk.Offset().DimensionIndices();
      });
      sum.at(sum_idx) = {output_layout, sum.at(sum_idx).Offset(),
                         chunk.Chunk()};
    }
    result = TOp::LaidOutTensorCt{sum};
  } else {
    const auto& resized = TResizeDimC(input_layout_, output_layout_);
    result = resized.AmendCtProgram(ct_program, {input_tensor});
  }

  Shape new_shape = output_layout.GetShape();
  const auto& old_shape = input_layout_.GetShape();

  // To minimize total ops, first replicate bits inside the chunk
  for (int bit_idx = ceil_log2(new_shape[dimension_]) - 1;
       bit_idx >= ceil_log2(old_shape[dimension_]); --bit_idx) {
    auto dim_bit = DimensionBit(dimension_, bit_idx);
    if (Estd::contains(output_layout_.Bits(), dim_bit)) {
      const auto& shift_bit = RawShiftBit(dim_bit, 1);
      result = DoRawShift(ct_program, result, shift_bit);
    }
  }
  // ...then replicate bits outside the chunk
  for (int bit_idx = ceil_log2(new_shape[dimension_]) - 1;
       bit_idx >= ceil_log2(old_shape[dimension_]); --bit_idx) {
    auto dim_bit = DimensionBit(dimension_, bit_idx);
    if (!Estd::contains(output_layout_.Bits(), dim_bit)) {
      const auto& shift_bit = RawShiftBit(dim_bit, 1);
      result = DoRawShift(ct_program, result, shift_bit);
    }
  }

  // No need to mask out invalid values if new dimension size is power of two
  if (IsPowerOfTwo(multiple_)) {
    return result;
  }
  return ApplyMask(ct_program, result, MaskAllInvalidSlots(result.Layout()));
}

bool TReplicateDimC::EqualTo(const TOp& other) const {
  const auto* t_replicate_dim_c = dynamic_cast<const TReplicateDimC*>(&other);
  return t_replicate_dim_c &&
         t_replicate_dim_c->OutputLayout() == OutputLayout() &&
         t_replicate_dim_c->InputLayout() == InputLayout() &&
         t_replicate_dim_c->DimensionToReplicate() == DimensionToReplicate() &&
         t_replicate_dim_c->Multiple() == Multiple();
}

}  // namespace fhelipe
