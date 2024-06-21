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

#ifndef FHELIPE_LAID_OUT_TENSOR_INDEX_H_
#define FHELIPE_LAID_OUT_TENSOR_INDEX_H_

#include "chunk_size.h"
#include "tensor_index.h"
#include "tensor_layout.h"

namespace fhelipe {
class TensorIndex;
class TensorLayout;

class LaidOutTensorIndex {
 public:
  LaidOutTensorIndex(const TensorLayout& layout, const TensorIndex& ti)
      : layout_(layout), ti_(ti) {}
  int ChunkNumber() const { return layout_.ChunkNumberAt(ti_); }
  int ChunkIndex() const { return layout_.ChunkIndexAt(ti_); }
  class ChunkSize ChunkSize() const { return layout_.ChunkSize(); }
  int TotalChunks() const { return layout_.TotalChunks(); }

 private:
  TensorLayout layout_;
  TensorIndex ti_;
};

class LaidOutTensorTranslation {
 public:
  LaidOutTensorTranslation(int num_chunks, class ChunkSize chunk_size,
                           int chunk_number_diff, int chunk_index_diff);
  int ChunkNumberDiff() const { return chunk_number_diff_; }
  int ChunkIndexDiff() const { return chunk_index_diff_; }
  class ChunkSize ChunkSize() const { return chunk_size_; }
  int TotalChunks() const { return num_chunks_; }

  friend bool operator==(const LaidOutTensorTranslation& lhs,
                         const LaidOutTensorTranslation& rhs);

 private:
  int num_chunks_;
  class ChunkSize chunk_size_;
  int chunk_number_diff_;
  int chunk_index_diff_;
};

inline bool operator==(const LaidOutTensorTranslation& lhs,
                       const LaidOutTensorTranslation& rhs) {
  return lhs.num_chunks_ == rhs.num_chunks_ &&
         lhs.chunk_size_ == rhs.chunk_size_ &&
         lhs.chunk_number_diff_ == rhs.chunk_number_diff_ &&
         lhs.chunk_index_diff_ == rhs.chunk_index_diff_;
}

LaidOutTensorIndex MakeLOTI(const TensorLayout& layout, const TensorIndex& ti);
LaidOutTensorTranslation TranslationSrcDest(const LaidOutTensorIndex& src,
                                            const LaidOutTensorIndex& dest);

}  // namespace fhelipe

#endif  // FHELIPE_LAID_OUT_TENSOR_INDEX_H_
