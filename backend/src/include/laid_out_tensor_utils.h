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

#ifndef FHELIPE_LAID_OUT_TENSOR_UTILS_H_
#define FHELIPE_LAID_OUT_TENSOR_UTILS_H_

#include <algorithm>
#include <iterator>
#include <vector>

#include "chunk.h"
#include "laid_out_tensor.h"
#include "t_op.h"
#include "tensor_index.h"
#include "tensor_layout.h"

namespace fhelipe {

template <typename SrcChunkType, typename DestChunkType>
LaidOutTensor<DestChunkType> Convert(
    const TensorLayout& layout, const std::vector<SrcChunkType>& in_chunks,
    std::function<DestChunkType(const SrcChunkType&)> convert_func) {
  std::vector<LaidOutChunk<DestChunkType>> out_chunks;
  for (int idx : Estd::indices(layout.ChunkOffsets().size())) {
    const auto& offset = layout.ChunkOffsets()[idx];
    const auto& chunk = convert_func(in_chunks[idx]);
    out_chunks.emplace_back(layout, offset, chunk);
  }
  return LaidOutTensor<DestChunkType>{out_chunks};
}

inline std::vector<TOp::LaidOutChunk> AdaptToLayout(
    const TensorLayout& output_layout,
    const std::vector<TOp::LaidOutChunk>& chunks) {
  CHECK(output_layout.TotalChunks() == chunks.size());
  return Estd::transform(
      chunks, [&output_layout](const TOp::LaidOutChunk& chunk) {
        return TOp::LaidOutChunk{
            output_layout,
            output_layout.ChunkOffsetAt(
                TensorIndex(output_layout.GetShape(), chunk.Offset().Flat())),
            chunk.Chunk()};
      });
}

template <typename SrcChunkType, typename DestChunkType, typename FuncType>
LaidOutTensor<DestChunkType> Convert(const LaidOutTensor<SrcChunkType>& tensor,
                                     FuncType convert_func) {
  auto chunks = Estd::transform(
      tensor.Chunks(), [](const auto& chunk) { return chunk.Chunk(); });
  return Convert<SrcChunkType, DestChunkType>(tensor.Layout(), chunks,
                                              convert_func);
}

}  // namespace fhelipe

#endif  // FHELIPE_LAID_OUT_TENSOR_UTILS_H_
