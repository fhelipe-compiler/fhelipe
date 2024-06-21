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

#ifndef FHELIPE_CHUNK_IR_H_
#define FHELIPE_CHUNK_IR_H_

#include <glog/logging.h>

#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "chunk_size.h"
#include "constants.h"
#include "dictionary.h"
#include "io_utils.h"
#include "plaintext_chunk.h"
#include "tensor.h"
#include "tensor_index.h"

namespace fhelipe {

class ZeroChunkIr {
 public:
  ZeroChunkIr(ChunkSize chunk_size) : chunk_size_(chunk_size) {}
  void WriteStreamHelper(std::ostream& stream) const { LOG(FATAL); }
  PtChunk Resolve(const Dictionary<Tensor<PtVal>>& frontend_tensors) const {
    return PtChunk(std::vector<PtVal>(chunk_size_.value(), 0));
  }

 private:
  ChunkSize chunk_size_;
};

class DirectChunkIr {
 public:
  explicit DirectChunkIr(const std::vector<PtVal>& values) : values_(values) {}
  PtChunk Resolve(const Dictionary<Tensor<PtVal>>& frontend_tensors) const;
  void WriteStreamHelper(std::ostream& stream) const;
  const std::vector<PtVal>& Values() const { return values_; }

 private:
  std::vector<PtVal> values_;
};

template <>
void WriteStream<DirectChunkIr>(std::ostream& stream, const DirectChunkIr& x);

template <>
DirectChunkIr ReadStream<DirectChunkIr>(std::istream& stream);

class IndirectChunkIr {
 public:
  IndirectChunkIr(const std::string& frontend_tensor_name,
                  const std::vector<std::optional<int>>& vec)
      : frontend_tensor_name_(frontend_tensor_name), flat_indices_(vec){};
  PtChunk Resolve(const Dictionary<Tensor<PtVal>>& frontend_tensors) const;
  void WriteStreamHelper(std::ostream& stream) const;
  const std::vector<std::optional<int>>& FlatIndices() const {
    return flat_indices_;
  }
  const std::string& FrontendTensorName() const {
    return frontend_tensor_name_;
  }

 private:
  std::string frontend_tensor_name_;
  std::vector<std::optional<int>> flat_indices_;
};

template <>
void WriteStream<IndirectChunkIr>(std::ostream& stream,
                                  const IndirectChunkIr& x);

template <>
IndirectChunkIr ReadStream<IndirectChunkIr>(std::istream& stream);

// nsamar: Using std::variant here because we want value semantics.
// Specifically, IndirectChunkIr and DirectChunkIr are different types;
// nonetheles, we want to keep them together in a container; so the only
// alternative to std::variant is inheritence. But inheritence has pointer
// semantics, which we don't want.
using ChunkIr = std::variant<IndirectChunkIr, DirectChunkIr, ZeroChunkIr>;

template <>
void WriteStream<ChunkIr>(std::ostream& stream, const ChunkIr& chunk);

template <>
ChunkIr ReadStream<ChunkIr>(std::istream& stream);

PtChunk Resolve(const ChunkIr& chunk,
                const Dictionary<Tensor<PtVal>>& frontend_tensors);

}  // namespace fhelipe

#endif  // FHELIPE_CHUNK_IR_H_
