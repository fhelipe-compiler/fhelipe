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

#include "include/packer.h"

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <utility>

#include "include/encryption_config.h"
#include "include/extended_std.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_index.h"
#include "include/maybe_tensor_index.h"
#include "include/plaintext.h"
#include "include/plaintext_chunk.h"
#include "include/shape.h"
#include "include/tensor.h"
#include "include/utils.h"

namespace fhelipe {

PtVal FlatIndexIntoLaidOutTensor(const LaidOutTensor<PtChunk>& lot,
                                 int flat_index) {
  auto ti = TensorIndex(lot.Layout().GetShape(), flat_index);
  return lot.AtOffset(lot.Layout().ChunkOffsetAt(ti))
      .Chunk()
      .Values()[lot.Layout().ChunkIndexAt(ti)];
}

Tensor<PtVal> Unpack(const LaidOutTensor<PtChunk>& tensor) {
  std::vector<PtVal> result =
      Estd::transform(Estd::indices(tensor.Layout().GetShape().ValueCnt()),
                      [&tensor](const auto& flat_index) {
                        return FlatIndexIntoLaidOutTensor(tensor, flat_index);
                      });

  return {tensor.Layout().GetShape(), result};
}

LaidOutChunk<PtChunk> IndexIntoVector(const std::vector<PtVal>& vec,
                                      const TensorLayout& layout,
                                      const TensorIndex& offset) {
  auto indices = layout.TensorIndices(offset);
  auto curr_chunk = Estd::transform(indices, [&vec](const auto& idx) {
    return idx.has_value() ? vec[idx.value().Flat()] : 0;
  });
  return {layout, offset, PtChunk(curr_chunk)};
}

LaidOutTensor<PtChunk> Pack(const std::vector<PtVal>& vec,
                            const TensorLayout& layout) {
  CHECK(layout.GetShape().ValueCnt() == vec.size());
  auto result = Estd::transform(layout.ChunkOffsets(),
                                [&vec, &layout](const auto& offset) {
                                  return IndexIntoVector(vec, layout, offset);
                                });
  return LaidOutTensor<PtChunk>{result};
}

}  // namespace fhelipe
