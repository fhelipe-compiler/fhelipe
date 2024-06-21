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

#include "include/chunk_ir.h"

#include "include/plaintext_chunk.h"

namespace fhelipe {

PtChunk DirectChunkIr::Resolve(
    const Dictionary<Tensor<PtVal>>& frontend_tensors) const {
  (void)frontend_tensors;
  return PtChunk(values_);
}

PtChunk IndirectChunkIr::Resolve(
    const Dictionary<Tensor<PtVal>>& frontend_tensors) const {
  auto frontend_tensor = frontend_tensors.At(frontend_tensor_name_);
  std::vector<PtVal> result;
  for (const auto& flat_idx : flat_indices_) {
    if (flat_idx.has_value()) {
      result.push_back(frontend_tensor[TensorIndex(frontend_tensor.GetShape(),
                                                   flat_idx.value())]);
    } else {
      result.push_back(0);
    }
  }
  return PtChunk(result);
}

template <>
void WriteStream<DirectChunkIr>(std::ostream& stream, const DirectChunkIr& x) {
  stream << kMaskChunkIrKeyword << " ";
  stream << x.Values().size() << " ";
  std::vector<int> results;
  for (int idx = 0; idx < x.Values().size(); ++idx) {
    if (x.Values()[idx] != 0) {
      CHECK(x.Values()[idx] == 1);
      results.push_back(idx);
    }
  }
  WriteStream(stream, results);
}

void DirectChunkIr::WriteStreamHelper(std::ostream& stream) const {
  WriteStream<DirectChunkIr>(stream, *this);
}

template <>
DirectChunkIr ReadStream<DirectChunkIr>(std::istream& stream) {
  auto chunk_size = ReadStream<int>(stream);
  std::vector<PtVal> values(chunk_size);
  auto ones = ReadStream<std::vector<int>>(stream);
  for (auto one : ones) {
    values.at(one) = 1;
  }
  return DirectChunkIr(values);
}

template <>
void WriteStream<IndirectChunkIr>(std::ostream& stream,
                                  const IndirectChunkIr& x) {
  stream << kIndirectChunkIrKeyword << " " << x.FrontendTensorName() << " ";
  WriteStream(stream, x.FlatIndices());
}

void IndirectChunkIr::WriteStreamHelper(std::ostream& stream) const {
  WriteStream<IndirectChunkIr>(stream, *this);
}

template <>
IndirectChunkIr ReadStream<IndirectChunkIr>(std::istream& stream) {
  auto frontend_tensor_name = ReadStream<std::string>(stream);
  auto vec = ReadStream<std::vector<std::optional<int>>>(stream);
  return {frontend_tensor_name, vec};
}

template <>
void WriteStream<ChunkIr>(std::ostream& stream, const ChunkIr& chunk) {
  std::visit([&](const auto& x) { x.WriteStreamHelper(stream); }, chunk);
}

template <>
ChunkIr ReadStream<ChunkIr>(std::istream& stream) {
  auto tensor_type = ReadStream<std::string>(stream);
  if (tensor_type == kIndirectChunkIrKeyword) {
    return ReadStream<IndirectChunkIr>(stream);
  }
  if (tensor_type == kMaskChunkIrKeyword) {
    return ReadStream<DirectChunkIr>(stream);
  }
  LOG(FATAL) << "Invalid ChunkIr type " << tensor_type;
}

PtChunk Resolve(const ChunkIr& chunk,
                const Dictionary<Tensor<PtVal>>& frontend_tensors) {
  return std::visit([&](const auto& x) { return x.Resolve(frontend_tensors); },
                    chunk);
}

}  // namespace fhelipe
