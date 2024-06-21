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

#include "include/laid_out_tensor_index.h"

#include <glog/logging.h>

#include <algorithm>

#include "include/chunk_size.h"
#include "include/tensor_layout.h"

namespace fhelipe {
class TensorIndex;
}  // namespace fhelipe

namespace {

int modulus(int a, int q) {
  CHECK(q > 0);
  while (a < 0) {
    a += q;
  }
  return a % q;
}

}  // namespace

namespace fhelipe {

LaidOutTensorTranslation::LaidOutTensorTranslation(int num_chunks,
                                                   class ChunkSize chunk_size,
                                                   int chunk_number_diff,
                                                   int chunk_index_diff)
    : num_chunks_(num_chunks),
      chunk_size_(chunk_size),
      chunk_number_diff_(modulus(chunk_number_diff, num_chunks)),
      chunk_index_diff_(modulus(chunk_index_diff, chunk_size.value())) {}

LaidOutTensorTranslation TranslationSrcDest(const LaidOutTensorIndex& src,
                                            const LaidOutTensorIndex& dest) {
  CHECK(src.ChunkSize() == dest.ChunkSize());
  int chunk_number_diff = dest.ChunkNumber() - src.ChunkNumber();
  int chunk_index_diff = dest.ChunkIndex() - src.ChunkIndex();
  return {std::max(src.TotalChunks(), dest.TotalChunks()), src.ChunkSize(),
          chunk_number_diff, chunk_index_diff};
}

}  // namespace fhelipe
