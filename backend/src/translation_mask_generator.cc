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

#include "include/translation_mask_generator.h"

#include <unordered_map>
#include <vector>

#include "include/laid_out_tensor_index.h"
#include "include/tensor_layout.h"

namespace fhelipe {

LaidOutTensor<ChunkIr> TranslationMaskGenerator::GetMask(
    const LaidOutTensorTranslation& diff) const {
  const auto& nonzeros = diff_map_.at(diff);

  std::unordered_map<TensorIndex, std::vector<PtVal>> result;
  for (const LaidOutTensorIndex& ti : nonzeros) {
    auto curr_offset = layout_.ChunkOffsets().at(ti.ChunkNumber());
    if (!result.contains(curr_offset)) {
      result.emplace(curr_offset,
                     std::vector<PtVal>(layout_.ChunkSize().value()));
    }
    result.at(curr_offset)[ti.ChunkIndex()] = 1;
  }

  std::vector<LaidOutChunk<ChunkIr>> mask_chunks;
  for (const auto& offset : layout_.ChunkOffsets()) {
    if (result.contains(offset)) {
      mask_chunks.emplace_back(layout_, offset,
                               DirectChunkIr(result.at(offset)));
    } else {
      mask_chunks.emplace_back(layout_, offset,
                               ZeroChunkIr(layout_.ChunkSize()));
    }
  }
  return LaidOutTensor<ChunkIr>{mask_chunks};
}

std::vector<TranslationMask> TranslationMaskGenerator::GetTranslationMasks()
    const {
  std::vector<TranslationMask> result;
  for (const auto& [key, value] : diff_map_) {
    result.emplace_back(key, GetMask(key));
  }
  return result;
}

void TranslationMaskGenerator::RegisterTranslation(
    const LaidOutTensorTranslation& diff, const LaidOutTensorIndex& ti) {
  diff_map_[diff].push_back(ti);
}

}  // namespace fhelipe
