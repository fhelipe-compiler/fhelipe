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

#include "include/chet_layout_pass.h"

#include <optional>
#include <vector>

#include "include/fill_gaps_layout_pass.h"
#include "include/layout_utils.h"
#include "include/t_chet_repack_c.h"
#include "include/t_replicate_dim_c.h"
#include "include/tensor_layout.h"

namespace fhelipe {

// Row-major
std::vector<TensorLayout::LayoutBit> ChetLayoutPass::DefaultLayoutBits(
    const Shape& shape) {
  std::vector<TensorLayout::LayoutBit> layout_bits;
  if (ROW_MAJOR_HACK) {
    for (int dim_idx : Estd::reverse(Estd::indices(shape.DimensionCount()))) {
      int dim_size = shape[dim_idx];
      for (int bit_idx = 0; bit_idx < ceil_log2(dim_size); ++bit_idx) {
        layout_bits.emplace_back(DimensionBit(dim_idx, bit_idx));
      }
    }
  } else {
    for (int dim_idx : Estd::reverse(Estd::indices(shape.DimensionCount()))) {
      int dim_size = shape[dim_idx];
      for (int bit_idx = 0; bit_idx < ceil_log2(dim_size); ++bit_idx) {
        layout_bits.emplace_back(DimensionBit(dim_idx, bit_idx));
      }
    }
  }
  return layout_bits;
}

std::vector<std::shared_ptr<Node<TOp>>> ChetLayoutPass::MatchLayouts(
    Dag<TOp>& dag, const std::vector<std::shared_ptr<Node<TOp>>>& parents) {
  if (AllLayoutsMatch(parents)) {
    return parents;
  }

  CHECK(parents.size() == 2);

  // Push all mismatching bits out to be chunk-selecting
  auto convert_to_me = Estd::transform(
      parents.at(0)->Value().OutputLayout().Bits(),
      parents.at(1)->Value().OutputLayout().Bits(),
      [](auto lhs_bit, auto rhs_bit) -> TensorLayout::LayoutBit {
        if (lhs_bit != rhs_bit) {
          return {std::nullopt};
        }
        return lhs_bit;
      });

  auto convert_to_me_t_op = TensorLayout(
      parents.at(0)->Value().OutputLayout().GetShape(), convert_to_me);
  auto lhs_converted =
      AddLayoutConversion(dag, parents.at(0), convert_to_me_t_op);
  auto rhs_converted =
      AddLayoutConversion(dag, parents.at(1), convert_to_me_t_op);
  return {lhs_converted, rhs_converted};
}

std::vector<TensorLayout::LayoutBit> ChetLayoutPass::ChetChunkBits(
    std::vector<TensorLayout::LayoutBit> layout_bits, ChunkSize chunk_size) {
  while (chunk_size.value() > (1 << layout_bits.size())) {
    layout_bits.push_back(std::nullopt);
  }
  int top_idx = ceil_log2(chunk_size.value());
  return std::vector<TensorLayout::LayoutBit>(layout_bits.begin(),
                                              layout_bits.begin() + top_idx);
}

TensorLayout ChetLayoutPass::DefaultLayout(const Shape& shape,
                                           ChunkSize chunk_size) const {
  auto layout_bits = DefaultLayoutBits(shape);
  return {shape, ChetChunkBits(layout_bits, chunk_size)};
}

TensorLayout ChetLayoutPass::GetTStrideCOutputLayout(
    const TensorLayout& input_layout, const std::vector<Stride>& strides) {
  return FillGapsLayoutPass::GetTStrideCOutputLayout(input_layout, strides);
}

TensorLayout ChetLayoutPass::GetTResizeDimsCOutputLayout(
    const TensorLayout& input_layout, const Shape& output_shape) {
  auto layout_bits = input_layout.Bits();
  for (auto& bit : layout_bits) {
    if (bit.has_value() &&
        output_shape[bit.value().dimension] <= (1 << bit.value().bit_index)) {
      bit = std::nullopt;
    }
  }
  return {output_shape, layout_bits};
}

TensorLayout ChetLayoutPass::GetTReduceDimCOutputLayout(
    const TensorLayout& input_layout, int dimension) {
  Shape new_shape =
      GetOutputShapeTReduceDimC(input_layout.GetShape(), dimension);
  new_shape[dimension] = 1;
  return GetTResizeDimsCOutputLayout(input_layout, new_shape);
}

TensorLayout ChetLayoutPass::GetTReplicateDimsCOutputLayout(
    const TensorLayout& input_layout, int dimension, int multiple) {
  Shape new_shape = GetOutputShapeTReplicateDimsC(input_layout.GetShape(),
                                                  dimension, multiple);
  auto resized_layout = GetTResizeDimsCOutputLayout(input_layout, new_shape);
  if (ChetLayoutPass::RowMajorHack()) {
    resized_layout =
        ChetRepackedLayout(LogChunkSize(input_layout.ChunkSize()), new_shape);
  }
  // No repacking on replicate
  return resized_layout;
}

TensorLayout ChetLayoutPass::GetTReorderDimsCOutputLayout(
    const TensorLayout& input_layout, const std::vector<int>& dim_order) {
  return FillGapsLayoutPass::GetTReorderDimsCOutputLayout(input_layout,
                                                          dim_order);
}

TensorLayout ChetLayoutPass::GetOutputLayout(
    const std::shared_ptr<Node<TOpEmbrio>>& embrio,
    const TensorLayout& input_layout) {
  if (const auto* t_reorder_dim_c =
          dynamic_cast<const TReorderDimsCEmbrio*>(&embrio->Value())) {
    return ChetLayoutPass::GetTReorderDimsCOutputLayout(
        input_layout, t_reorder_dim_c->DimensionOrder());
  }
  if (const auto* t_replicate_dim_c =
          dynamic_cast<const TReplicateDimCEmbrio*>(&embrio->Value())) {
    return ChetLayoutPass::GetTReplicateDimsCOutputLayout(
        input_layout, t_replicate_dim_c->DimensionToReplicate(),
        t_replicate_dim_c->ReplicationMultiple());
  }
  if (const auto* t_reduce_dim_c =
          dynamic_cast<const TReduceDimCEmbrio*>(&embrio->Value())) {
    return ChetLayoutPass::GetTReduceDimCOutputLayout(
        input_layout, t_reduce_dim_c->DimensionToReduce());
  }
  if (const auto* t_stride_c =
          dynamic_cast<const TStrideCEmbrio*>(&embrio->Value())) {
    return ChetLayoutPass::GetTStrideCOutputLayout(input_layout,
                                                   t_stride_c->Strides());
  }
  if (const auto* t_stride_c =
          dynamic_cast<const TMergedStrideCEmbrio*>(&embrio->Value())) {
    return ChetLayoutPass::GetTStrideCOutputLayout(input_layout,
                                                   t_stride_c->Strides());
  }
  if (const auto* t_resize_dim_c =
          dynamic_cast<const TResizeDimCEmbrio*>(&embrio->Value())) {
    return ChetLayoutPass::GetTResizeDimsCOutputLayout(
        input_layout, embrio->Value().OutputShape());
  }
  if (const auto* t_chet_repack_c =
          dynamic_cast<const TChetRepackCEmbrio*>(&embrio->Value())) {
    return ChetRepackedLayout(LogChunkSize(input_layout.ChunkSize()),
                              input_layout.GetShape());
  }
  return input_layout;
}

}  // namespace fhelipe
