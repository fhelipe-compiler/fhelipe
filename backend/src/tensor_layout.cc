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

#include "include/tensor_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <bitset>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

#include "include/array.h"
#include "include/dimension_bit.h"
#include "include/extended_std.h"
#include "include/index_mask.h"
#include "include/io_utils.h"
#include "include/maybe_tensor_index.h"
#include "include/shape.h"
#include "include/tensor_index.h"

namespace fhelipe {

std::vector<DimensionBit> TensorLayout::TensorOffsetBits() const {
  std::vector<DimensionBit> result;

  for (int dim_i = shape_.DimensionCount() - 1; dim_i >= 0; --dim_i) {
    IndexMask full_mask = MaxIndexMask(shape_[dim_i]);
    IndexMask mask_in_chunk = MaskOfDimension(dim_i);
    IndexMask mask_in_offset = full_mask & (~mask_in_chunk);
    std::vector<int> bit_inds = MaskedIndices(mask_in_offset);

    auto append_me = Estd::transform(
        bit_inds, [&dim_i](int bit_i) { return DimensionBit(dim_i, bit_i); });
    Estd::append(result, append_me);
  }
  return result;
}

int TensorLayout::ChunkNumberAt(const TensorIndex& ti) const {
  int chunk_offset = ChunkOffsetAt(ti).Flat();
  const auto& offsets = ChunkOffsets();
  // Binary search
  int low = 0;
  int high = offsets.size();
  int mid = (low + high) / 2;
  while (offsets[mid].Flat() != chunk_offset) {
    if (offsets[mid].Flat() > chunk_offset) {
      high = mid;
    } else {
      low = mid;
    }
    mid = (low + high) / 2;
  }
  return mid;
}

TensorIndex TensorLayout::ChunkOffsetAt(const TensorIndex& ti) const {
  std::vector<int> result(ti.DimensionIndices().size());
  auto offset_bits = TensorOffsetBits();
  auto ti_indices = ti.DimensionIndices();
  for (int dim = 0; dim < ti.GetShape().DimensionCount(); ++dim) {
    int ti_idx = ti_indices[dim];
    for (int bit_idx = 0; bit_idx < offset_bits.size(); ++bit_idx) {
      TensorLayout::LayoutBit lb = offset_bits[bit_idx];
      if (!lb.has_value()) {
        continue;
      }
      if (lb.value().dimension == dim) {
        if ((1 << lb.value().bit_index) & ti_idx) {
          result[dim] += (1 << lb.value().bit_index);
        }
      }
    }
  }
  return TensorIndex(GetShape(), result);
}

static std::unordered_map<TensorLayout, std::vector<TensorIndex>> chunk_offsets;

const std::vector<TensorIndex>& TensorLayout::ChunkOffsets() const {
  if (Estd::contains_key(chunk_offsets, *this)) {
    CHECK(chunk_offsets.at(*this)[0].GetShape() == GetShape());
    return chunk_offsets.at(*this);
  }

  const auto offset_bits = TensorOffsetBits();

  int offset_count = 1 << offset_bits.size();
  std::vector<TensorIndex> offsets;
  offsets.reserve(offset_count);

  for (int i = 0; i < offset_count; ++i) {
    IndexMask offset_mask(i);

    std::vector<int> dim_indices(shape_.DimensionCount(), 0);
    for (int j : MaskedIndices(offset_mask)) {
      const auto& [dimension_i, bit_i] = offset_bits[j];
      dim_indices[dimension_i] += 1 << bit_i;
    }
    if (IsInRange(shape_, Array(dim_indices))) {
      offsets.emplace_back(shape_, dim_indices);
    }
  }

  chunk_offsets.emplace(*this, offsets);
  return chunk_offsets.at(*this);
}

MaybeTensorIndex TensorLayout::TensorIndexAt(int index) const {
  std::vector<int> dim_inds(shape_.DimensionCount(), 0);

  for (const auto& [bit_i, bit] : BitsInIndex(index)) {
    if (!bit.has_value()) {
      return std::nullopt;
    }
    dim_inds[bit->dimension] += (1 << bit->bit_index);
  }
  for (const auto& [bit_i, bit] : BitsInIndex(index)) {
    if (dim_inds[bit->dimension] >= shape_[bit->dimension]) {
      return std::nullopt;
    }
  }
  return {TensorIndex(shape_, dim_inds)};
}

int TensorLayout::ChunkIndexAt(const TensorIndex& ti) const {
  int chunk_idx = 0;
  auto ti_indices = ti.DimensionIndices();
  for (int dim = 0; dim < ti.GetShape().DimensionCount(); ++dim) {
    int ti_idx = ti_indices[dim];
    for (int bit_idx = 0; bit_idx < bits_.size(); ++bit_idx) {
      LayoutBit lb = bits_[bit_idx];
      if (!lb.has_value()) {
        continue;
      }
      if (lb.value().dimension == dim) {
        if ((1 << lb.value().bit_index) & ti_idx) {
          chunk_idx += (1 << bit_idx);
        }
      }
    }
  }
  return chunk_idx;
}

// Performance optimization
std::unordered_map<
    TensorLayout,
    std::unordered_map<TensorIndex, std::vector<MaybeTensorIndex>>>
    tensor_indices_cache;

std::vector<MaybeTensorIndex> TensorLayout::TensorIndices(
    TensorIndex offset) const {
  CHECK(shape_ == offset.GetShape());
  if (Estd::contains_key(tensor_indices_cache, *this) &&
      Estd::contains_key(tensor_indices_cache.at(*this), offset)) {
    return tensor_indices_cache.at(*this).at(offset);
  }

  std::vector<MaybeTensorIndex> result;
  result.reserve(ChunkSize().value());
  for (int i = 0; i < ChunkSize().value(); ++i) {
    MaybeTensorIndex raw_result = TensorIndexAt(i);
    if (raw_result.has_value()) {
      Array dim_indices = raw_result.value().DimensionIndices();
      Array offset_indices = offset.DimensionIndices();
      dim_indices = Estd::transform(dim_indices, offset_indices, std::plus<>());
      if (IsInRange(shape_, dim_indices)) {
        result.emplace_back(TensorIndex(shape_, dim_indices));
      } else {
        result.emplace_back(std::nullopt);
      }
    } else {
      result.push_back(raw_result);
    }
  }
  tensor_indices_cache[*this].emplace(offset, result);
  return result;
}

IndexMask TensorLayout::MaskOfChunk() const {
  IndexMask mask;
  for (int i = 0; i < bits_.size(); ++i) {
    mask[i] = bits_[i].has_value();
  }
  return mask;
}

IndexMask TensorLayout::MaskOfDimension(int dimension) const {
  IndexMask mask;
  for (int i = 0; i < bits_.size(); ++i) {
    const LayoutBit& bit = bits_[i];
    if (bit.has_value() && bit->dimension == dimension) {
      mask[bit->bit_index] = true;
    }
  }
  return mask;
}

void TensorLayout::CheckRep() const {
  for (int i = 0; i < bits_.size(); ++i) {
    for (int j = i + 1; j < bits_.size(); ++j) {
      CHECK(bits_[i] != bits_[j] || !bits_[i].has_value());
    }
  }
  for (const auto& bit : bits_) {
    if (bit.has_value()) {
      CHECK(shape_.DimensionCount() > bit.value().dimension);
      CHECK(shape_[bit.value().dimension] >= (1 << bit.value().bit_index));
    }
  }
}

std::vector<std::pair<int, TensorLayout::LayoutBit>> TensorLayout::BitsInIndex(
    int index) const {
  std::vector<int> indices = MaskedIndices(IndexMask(index));
  return Estd::transform(
      indices, [this](int index) -> std::pair<int, TensorLayout::LayoutBit> {
        return std::make_pair(index, this->bits_[index]);
      });
}

TensorLayout::TensorLayout(const Shape& shape,
                           const std::vector<LayoutBit>& layout_bits)
    : shape_(shape), bits_(layout_bits) {
  CheckRep();
}

std::string PrettyPrint(const std::vector<TensorLayout::LayoutBit>& value) {
  std::stringstream ss;
  ss << "idx: ";
  auto indices = Estd::reverse(Estd::indices(value.size()));
  auto bits = Estd::reverse(value);
  for (int idx : indices) {
    ss << std::setfill(' ') << std::setw(2) << idx << " ";
  }
  ss << "\n";
  ss << "dim: ";
  for (const auto& bit : bits) {
    if (bit.has_value()) {
      ss << std::setfill(' ') << std::setw(2) << bit.value().dimension << " ";
    } else {
      ss << " # ";
    }
  }
  ss << "\n";
  ss << "bit: ";
  for (const auto& bit : bits) {
    if (bit.has_value()) {
      ss << std::setfill(' ') << std::setw(2) << bit.value().bit_index << " ";
    } else {
      ss << " # ";
    }
  }
  return ss.str();
}

template <>
std::string DagLabel<std::vector<TensorLayout::LayoutBit>>(
    const std::vector<TensorLayout::LayoutBit>& value) {
  std::stringstream ss;
  ss << "idx: ";
  auto indices = Estd::reverse(Estd::indices(value.size()));
  auto bits = Estd::reverse(value);
  for (int idx : indices) {
    ss << std::setfill(' ') << std::setw(2) << idx << " ";
  }
  ss << "\\n";
  ss << "dim: ";
  for (const auto& bit : bits) {
    if (bit.has_value()) {
      ss << std::setfill(' ') << std::setw(2) << bit.value().dimension << " ";
    } else {
      ss << " # ";
    }
  }
  ss << "\\n";
  ss << "bit: ";
  for (const auto& bit : bits) {
    if (bit.has_value()) {
      ss << std::setfill(' ') << std::setw(2) << bit.value().bit_index << " ";
    } else {
      ss << " # ";
    }
  }
  return ss.str();
}
template <>
std::string DagLabel<TensorLayout>(const TensorLayout& value) {
  std::stringstream ss;
  ss << "shape: ";
  for (int dim : value.GetShape()) {
    ss << dim << " ";
  }
  ss << std::endl;
  ss << DagLabel(value.Bits());
  return ss.str();
}

}  // namespace fhelipe
