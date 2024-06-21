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

#include "include/t_unpadded_shift_c.h"

#include <glog/logging.h>

#include <optional>
#include <string>

#include "include/ct_program.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_index.h"
#include "include/shape.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_generator.h"
#include "include/translation_mask_utils.h"

namespace fhelipe {

TOp::LaidOutTensorCt TUnpaddedShiftC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  const auto& input_tensor = input_tensors[0];
  CHECK(input_tensor.Layout() == layout_);
  const auto& input_layout = input_tensor.Layout();
  return (BackendMaskDepth() > 0)
             ? TOp::LaidOutTensorCt{ApplyTranslationMasks(
                   ct_program, input_tensor, translation_masks_, input_layout)}
             : TOp::LaidOutTensorCt{ApplyTranslationsButNotMasks(
                   ct_program, input_tensor, translation_masks_, input_layout)};
}

void TUnpaddedShiftC::SetLayouts(const TensorLayout& input_layout,
                                 const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
  translation_masks_ = MakeTranslationMasks(
      layout_, OutputLayout(),
      [this](const TensorIndex& ti) { return rotate_by_.NonCyclicAdd(ti); });
}

bool TUnpaddedShiftC::EqualTo(const TOp& other) const {
  const auto* t_unpadded_shift_c = dynamic_cast<const TUnpaddedShiftC*>(&other);
  return t_unpadded_shift_c &&
         t_unpadded_shift_c->OutputLayout() == OutputLayout() &&
         RotateBy() == t_unpadded_shift_c->RotateBy();
}

TUnpaddedShiftC::TUnpaddedShiftC(const TensorLayout& layout,
                                 const DiffTensorIndex& rotate_by)
    : layout_(layout),
      rotate_by_(rotate_by),
      translation_masks_(MakeTranslationMasks(
          layout_, OutputLayout(), [this](const TensorIndex& ti) {
            return rotate_by_.NonCyclicAdd(ti);
          })) {}

int TUnpaddedShiftC::BackendMaskDepth() const {
  // nsamar: If all translation masks are either:
  // 1) all zeroes
  // OR
  // 2) only have 1s at the valid slots, and 0s in invalid slots, then the mask
  // need not be applied, because the invalid slots are already zeroed out!
  for (const auto& [translation, mask_tensor] : translation_masks_) {
    int chunk_number = 0;
    for (const auto& chunk : mask_tensor.Chunks()) {
      if (std::holds_alternative<ZeroChunkIr>(chunk.Chunk())) {
        continue;
      }
      const auto& values = std::get<DirectChunkIr>(chunk.Chunk()).Values();
      auto indices =
          layout_.TensorIndices(layout_.ChunkOffsets()[chunk_number]);
      for (int i = 0; i < values.size(); i++) {
        if (indices[i].has_value() && values[i] == 0) {
          // What would happen if a value that wasn't selected, was selected...
          // if we end up in a index that is not to be padded, then we still
          // don't need to mask!
          int dest_chunk = chunk_number + translation.ChunkNumberDiff();
          int slot =
              (i + translation.ChunkIndexDiff()) % layout_.ChunkSize().value();
          auto dest_ti =
              layout_.TensorIndices(layout_.ChunkOffsets()[dest_chunk])[slot];
          if (!dest_ti.has_value()) {
            return 1;
          }
          for (int dim_idx = 0; dim_idx < layout_.GetShape().DimensionCount();
               dim_idx++) {
            int target_idx = dest_ti.value()[dim_idx] - rotate_by_[dim_idx];
            if (target_idx < 0 || target_idx >= layout_.GetShape()[dim_idx]) {
              goto notit;
            }
          }
          return 1;
        notit:;
        }
      }
      chunk_number++;
    }
  }
  return 0;
}

}  // namespace fhelipe
