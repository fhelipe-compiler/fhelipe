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

#ifndef FHELIPE_TRANSLATION_MASK_GENERATOR_H_
#define FHELIPE_TRANSLATION_MASK_GENERATOR_H_

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chunk_ir.h"
#include "laid_out_tensor.h"
#include "laid_out_tensor_index.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

struct transhash {
 public:
  std::size_t operator()(const LaidOutTensorTranslation& x) const {
    return std::hash<int>()(x.ChunkNumberDiff()) ^
           std::hash<int>()(x.ChunkIndexDiff());
  }
};

LaidOutTensor<ChunkIr> MaskAllInvalidSlots(const TensorLayout& layout);

TOp::LaidOutTensorCt ApplyMask(ct_program::CtProgram& ct_program,
                               const TOp::LaidOutTensorCt& ct,
                               const LaidOutTensor<ChunkIr>& pt);

using TranslationMask =
    std::pair<LaidOutTensorTranslation, LaidOutTensor<ChunkIr>>;

class TranslationMaskGenerator {
 public:
  explicit TranslationMaskGenerator(const TensorLayout& layout)
      : layout_(layout) {}

  std::vector<TranslationMask> GetTranslationMasks() const;
  void RegisterTranslation(const LaidOutTensorTranslation& diff,
                           const LaidOutTensorIndex& ti);

 private:
  std::unordered_map<LaidOutTensorTranslation, std::vector<LaidOutTensorIndex>,
                     transhash>
      diff_map_;
  TensorLayout layout_;
  LaidOutTensor<ChunkIr> GetMask(const LaidOutTensorTranslation& trans) const;
};

}  // namespace fhelipe

#endif  // FHELIPE_TRANSLATION_MASK_GENERATOR_H_
