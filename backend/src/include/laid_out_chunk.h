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

#ifndef FHELIPE_LAID_OUT_CHUNK_H_
#define FHELIPE_LAID_OUT_CHUNK_H_

#include "include/chunk.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"

namespace fhelipe {

template <class T>
class LaidOutChunk {
 public:
  LaidOutChunk(const TensorLayout& layout, const TensorIndex& offset,
               const T& chunk)
      : layout_(layout), offset_(offset), chunk_(chunk) {
    CHECK(offset_.GetShape() == layout_.GetShape());
    CHECK(Estd::contains(layout.ChunkOffsets(), offset_));
  }

  const TensorLayout& Layout() const { return layout_; }
  const TensorIndex& Offset() const { return offset_; }
  const T& Chunk() const { return chunk_; }
  T& Chunk() { return chunk_; }

 private:
  TensorLayout layout_;
  TensorIndex offset_;
  T chunk_;
};

template <typename T>
bool operator<(const LaidOutChunk<T>& lhs, const LaidOutChunk<T>& rhs) {
  CHECK(lhs.Layout() == rhs.Layout());
  return lhs.Offset() < rhs.Offset();
}

}  // namespace fhelipe

#endif  // FHELIPE_LAID_OUT_CHUNK_H_
