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

#include "include/t_reduce_dim_c.h"

#include <glog/logging.h>

#include "include/ct_program.h"
#include "include/dimension_bit.h"
#include "include/laid_out_tensor.h"
#include "include/raw_shift_acc.h"
#include "include/raw_shift_bit.h"
#include "include/shape.h"
#include "include/t_resize_dim_c.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_generator.h"
#include "include/utils.h"

namespace fhelipe {

class CtOp;

Shape GetOutputShapeTReduceDimC(Shape shape, int dimension) {
  shape[dimension] = 1;
  return shape;
}

void SanityCheckTReduceDimC(const TensorLayout& input_layout,
                            const TensorLayout& output_layout, int dimension) {
  CHECK(dimension >= 0 && dimension < input_layout.GetShape().DimensionCount());
  CHECK(output_layout.GetShape()[dimension] == 1);
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

TReduceDimC::TReduceDimC(const TensorLayout& input_layout,
                         const TensorLayout& output_layout, int dimension)
    : input_layout_(input_layout),
      output_layout_(output_layout),
      dimension_(dimension) {
  SanityCheckTReduceDimC(input_layout, output_layout, dimension);
}

void TReduceDimC::SetLayouts(const TensorLayout& input_layout,
                             const TensorLayout& output_layout) {
  SanityCheckTReduceDimC(input_layout, output_layout, dimension_);
  input_layout_ = input_layout;
  output_layout_ = output_layout;
}

TOp::LaidOutTensorCt TReduceDimC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  const auto& input_tensor = input_tensors[0];
  CHECK(input_tensor.Layout() == input_layout_);
  (void)ct_program;
  const Shape& shape = input_layout_.GetShape();
  auto result = input_tensor;

  // To minimize total ops, first reduce bits outside chunks
  for (int bit_idx = ceil_log2(shape[dimension_]) - 1; bit_idx > -1;
       --bit_idx) {
    auto dim_bit = DimensionBit(dimension_, bit_idx);
    if (!Estd::contains(input_layout_.Bits(), dim_bit)) {
      const auto& shift_bit = RawShiftBit(dim_bit, -1);
      result = DoRawShift(ct_program, result, shift_bit);
      for (int idx : Estd::indices(result.Chunks().size())) {
        if (!WrapsAround(shift_bit, result.Chunks().at(idx).Offset())) {
          // Make redundant chunks all zeroes so that we can avoid redundant
          // rotates inside chunks later on
          result.Chunks().at(idx).Chunk() =
              ct_program::FetchZeroCAtSameLevelInfoAs(
                  result.Chunks().at(idx).Chunk());
        }
      }
    }
  }

  // ...then reduce bits inside chunks
  for (int bit_idx = ceil_log2(shape[dimension_]) - 1; bit_idx > -1;
       --bit_idx) {
    auto dim_bit = DimensionBit(dimension_, bit_idx);
    if (Estd::contains(input_layout_.Bits(), dim_bit)) {
      const auto& shift_bit = RawShiftBit(dim_bit, -1);
      result = DoRawShift(ct_program, result, shift_bit);
    }
  }

  const auto& resized = TResizeDimC(input_layout_, output_layout_);
  return resized.AmendCtProgram(ct_program, {result});
}

bool TReduceDimC::EqualTo(const TOp& other) const {
  const auto* t_reduce_dim_c = dynamic_cast<const TReduceDimC*>(&other);
  return t_reduce_dim_c && t_reduce_dim_c->OutputLayout() == OutputLayout() &&
         t_reduce_dim_c->InputLayout() == InputLayout() &&
         DimensionToReduce() == t_reduce_dim_c->DimensionToReduce();
}

}  // namespace fhelipe
