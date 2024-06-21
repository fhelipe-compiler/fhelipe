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

#ifndef FHELIPE_LAID_OUT_TENSOR_H_
#define FHELIPE_LAID_OUT_TENSOR_H_

#include <glog/logging.h>

#include <algorithm>
#include <vector>

#include "chunk_ir.h"
#include "ct_op.h"
#include "laid_out_chunk.h"
#include "plaintext_chunk.h"
#include "tensor_index.h"
#include "tensor_layout.h"

namespace fhelipe {

namespace detail {

template <class T>
void CheckLevelInfo(const std::vector<LaidOutChunk<T>>& locs) = delete;

template <>
inline void CheckLevelInfo<PtChunk>(
    const std::vector<LaidOutChunk<PtChunk>>& locs) {  // nothing to check
}

template <>
inline void CheckLevelInfo<ChunkIr>(
    const std::vector<LaidOutChunk<ChunkIr>>& locs) {  // nothing to check
}

}  // namespace detail

template <class T>
class LaidOutTensor {
 public:
  explicit LaidOutTensor(const std::vector<LaidOutChunk<T>>& chunks);

  // nsamar: LaidOutTensor guarantees that LaidOutChunk returns chunks in
  // increasing order of flat index when Chunks() is called
  const std::vector<LaidOutChunk<T>>& Chunks() const;
  std::vector<LaidOutChunk<T>>& Chunks();
  int Slots() const { return 1 << layout_.Bits().size(); }
  const TensorLayout& Layout() const;
  std::vector<TensorIndex> Offsets() const;
  const LaidOutChunk<T>& AtOffset(const TensorIndex& offset) const;

 private:
  TensorLayout layout_;
  std::vector<LaidOutChunk<T>> chunks_;
};

template <class T>
bool operator==(const LaidOutTensor<T>& lhs, const LaidOutTensor<T>& rhs) {
  return lhs.Layout() == rhs.Layout() && lhs.chunks_() == rhs.chunks_();
}

template <class T>
LaidOutTensor<T>::LaidOutTensor(const std::vector<LaidOutChunk<T>>& chunks)
    : layout_(chunks[0].Layout()), chunks_(chunks) {
  CHECK(chunks_.size() == layout_.TotalChunks());
  std::vector<TensorIndex> chunk_offsets = Estd::transform(
      chunks_, [](const auto& chunk) { return chunk.Offset(); });
  CHECK(Estd::is_equal_as_sets(chunk_offsets, layout_.ChunkOffsets()));
  Estd::for_each(
      chunks_, [this](const auto chunk) { CHECK(chunk.Layout() == layout_); });
  std::sort(chunks_.begin(), chunks_.end());
  detail::CheckLevelInfo<T>(chunks_);
}

template <class T>
inline const std::vector<LaidOutChunk<T>>& LaidOutTensor<T>::Chunks() const {
  return chunks_;
}

template <class T>
inline std::vector<LaidOutChunk<T>>& LaidOutTensor<T>::Chunks() {
  return chunks_;
}

template <class T>
inline const TensorLayout& LaidOutTensor<T>::Layout() const {
  return layout_;
}

template <class T>
inline std::vector<TensorIndex> LaidOutTensor<T>::Offsets() const {
  return layout_.ChunkOffsets();
}

template <class T>
const LaidOutChunk<T>& LaidOutTensor<T>::AtOffset(
    const TensorIndex& offset) const {
  CHECK(Estd::contains(layout_.ChunkOffsets(), offset));
  for (const auto& chunk : chunks_) {
    if (chunk.Offset() == offset) {
      return chunk;
    }
  }
  LOG(FATAL);
}

}  // namespace fhelipe

#endif  // FHELIPE_LAID_OUT_TENSOR_H_
