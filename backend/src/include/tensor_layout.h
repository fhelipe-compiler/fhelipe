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

#ifndef FHELIPE_TENSOR_LAYOUT_H_
#define FHELIPE_TENSOR_LAYOUT_H_

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "constants.h"
#include "dimension_bit.h"
#include "index_mask.h"
#include "io_utils.h"
#include "maybe_tensor_index.h"
#include "shape.h"
#include "tensor_index.h"

namespace fhelipe {

class TensorLayout {
 public:
  using LayoutBit = std::optional<DimensionBit>;
  TensorLayout(const Shape& shape, const std::vector<LayoutBit>& layout_bits);

  const Shape& GetShape() const;
  TensorIndex ChunkOffsetAt(const TensorIndex& ti) const;
  int TotalChunks() const;
  int ChunkNumberAt(const TensorIndex& ti) const;

  const std::vector<TensorIndex>& ChunkOffsets() const;
  class ChunkSize ChunkSize() const;

  MaybeTensorIndex TensorIndexAt(int chunk_index) const;
  std::vector<MaybeTensorIndex> TensorIndices(TensorIndex offset) const;
  int ChunkIndexAt(const TensorIndex& ti) const;
  IndexMask MaskOfChunk() const;
  IndexMask MaskOfDimension(int dimension) const;

  const std::vector<LayoutBit>& Bits() const { return bits_; }
  friend bool operator==(const TensorLayout& rhs, const TensorLayout& lhs);
  std::vector<DimensionBit> TensorOffsetBits() const;

 private:
  Shape shape_;
  std::vector<LayoutBit> bits_;

  void CheckRep() const;
  std::vector<std::pair<int, LayoutBit>> BitsInIndex(int index) const;
};

inline const Shape& TensorLayout::GetShape() const { return shape_; }

inline int TensorLayout::TotalChunks() const { return ChunkOffsets().size(); }

inline bool operator==(const TensorLayout& lhs, const TensorLayout& rhs) {
  return lhs.bits_ == rhs.bits_ && lhs.shape_ == rhs.shape_;
}

inline bool operator!=(const TensorLayout& lhs, const TensorLayout& rhs) {
  return !(lhs == rhs);
}

inline class ChunkSize TensorLayout::ChunkSize() const {
  return 1 << bits_.size();
}

template <>
inline void WriteStream<TensorLayout::LayoutBit>(
    std::ostream& stream, const TensorLayout::LayoutBit& lb) {
  if (lb.has_value()) {
    WriteStream(stream, lb.value());
  } else {
    stream << kInvalidOptionalToken;
  }
}

template <>
inline TensorLayout::LayoutBit ReadStream<TensorLayout::LayoutBit>(
    std::istream& stream) {
  auto token = ReadStream<std::string>(stream);
  if (token == kInvalidOptionalToken) {
    return std::nullopt;
  }
  auto bit_index = ReadStream<int>(stream);
  return std::make_optional(DimensionBit(std::stoi(token), bit_index));
}

template <>
inline void WriteStream<TensorLayout>(std::ostream& stream,
                                      const TensorLayout& layout) {
  Shape shape = layout.GetShape();
  WriteStream(stream, shape);
  stream << " ";
  WriteStream(stream, layout.Bits());
}

template <>
inline TensorLayout ReadStream<TensorLayout>(std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto bits = ReadStream<std::vector<TensorLayout::LayoutBit>>(stream);
  return {shape, bits};
}

template <>
std::string DagLabel<TensorLayout>(const TensorLayout& value);

template <>
std::string DagLabel<std::vector<TensorLayout::LayoutBit>>(
    const std::vector<TensorLayout::LayoutBit>& value);

std::string PrettyPrint(const std::vector<TensorLayout::LayoutBit>& value);

}  // namespace fhelipe

template <>
struct std::hash<fhelipe::TensorLayout> {
  std::size_t operator()(const fhelipe::TensorLayout& layout) const noexcept {
    std::size_t seed = std::hash<size_t>()(layout.Bits().size());
    for (const auto& bit : layout.Bits()) {
      int x = 0;
      if (bit.has_value()) {
        x = bit.value().dimension;
      } else {
        x = -1;
      }
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = (x >> 16) ^ x;
      seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      if (bit.has_value()) {
        x = bit.value().bit_index;
      } else {
        x = -1;
      }
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = (x >> 16) ^ x;
      seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed ^ std::hash<fhelipe::Shape>()(layout.GetShape());
  }
};

#endif  // FHELIPE_TENSOR_LAYOUT_H_
