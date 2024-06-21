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

#include "include/raw_shift_acc.h"

#include <algorithm>
#include <vector>

#include "include/dimension_bit.h"
#include "include/laid_out_tensor.h"
#include "include/raw_shift_bit.h"
#include "include/shape.h"
#include "include/t_add_cc.h"
#include "include/t_op.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_utils.h"

namespace fhelipe {
namespace ct_program {
class CtProgram;
}  // namespace ct_program

bool WrapsAround(const RawShiftBit& shift_bit, const TensorIndex& offset) {
  const Shape& shape = offset.GetShape();
  int result_dim_idx = offset[shift_bit.Dimension()] + shift_bit.ShiftAmount();
  return (result_dim_idx < 0 || result_dim_idx >= shape[shift_bit.Dimension()]);
}

namespace {

bool IsRawShiftInChunk(const TensorLayout& layout,
                       const RawShiftBit& shift_bit) {
  const auto& layout_bits = layout.Bits();
  const auto& relevant_bit = std::find(layout_bits.begin(), layout_bits.end(),
                                       shift_bit.GetDimensionBit());
  return (relevant_bit != layout_bits.end());
}

TOp::LaidOutTensorCt RawShiftedChunks(
    const std::vector<TOp::LaidOutChunk>& chunks,
    const RawShiftBit& shift_bit) {
  const auto& layout = chunks[0].Layout();
  if (IsRawShiftInChunk(layout, shift_bit)) {
    return TOp::LaidOutTensorCt{chunks};
  }

  const DiffTensorIndex& rotate_diff = shift_bit.ShiftDiff(layout.GetShape());

  std::unordered_map<TensorIndex, TOp::LaidOutChunk> result;
  auto zero_c = ct_program::FetchZeroCAtSameLevelInfoAs(chunks.at(0).Chunk());
  for (const auto& chunk : chunks) {
    result.emplace(chunk.Offset(),
                   TOp::LaidOutChunk{layout, chunk.Offset(), zero_c});
  }
  for (const auto& chunk : chunks) {
    if (!WrapsAround(shift_bit, chunk.Offset())) {
      auto new_offset =
          layout.ChunkOffsetAt(rotate_diff.CyclicAdd(chunk.Offset()));
      result.at(new_offset) =
          TOp::LaidOutChunk{layout, new_offset, chunk.Chunk()};
    }
  }

  return TOp::LaidOutTensorCt{Estd::get_values(result)};
}

int GetRotateBy(const TensorLayout& layout, const RawShiftBit& shift_bit) {
  const auto& layout_bits = layout.Bits();
  int log2_rotate_by = std::find(layout_bits.begin(), layout_bits.end(),
                                 shift_bit.GetDimensionBit()) -
                       layout_bits.begin();
  int rotate_by = shift_bit.Direction() * (1 << log2_rotate_by);
  return rotate_by;
}

std::vector<TOp::LaidOutChunk> RotateChunks(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutChunk>& chunks,
    const RawShiftBit& shift_bit) {
  int rotate_by = GetRotateBy(chunks[0].Layout(), shift_bit);
  return ApplyRotation(ct_program, chunks, rotate_by);
}

}  // namespace

TOp::LaidOutTensorCt DoRawShift(ct_program::CtProgram& ct_program,
                                const TOp::LaidOutTensorCt& input_tensor,
                                const RawShiftBit& shift_bit) {
  const auto& rotated =
      RotateChunks(ct_program, input_tensor.Chunks(), shift_bit);
  const auto& shuffled_chunks = RawShiftedChunks(rotated, shift_bit);
  const auto& add_cc = TAddCC(input_tensor.Layout());
  return add_cc.AmendCtProgram(ct_program, {shuffled_chunks, input_tensor});
}

}  // namespace fhelipe
