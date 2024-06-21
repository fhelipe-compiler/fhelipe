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

#include "include/fill_gaps_layout_pass.h"

#include <glog/logging.h>

#include <optional>
#include <unordered_map>
#include <vector>

#include "include/chunk_size.h"
#include "include/dag.h"
#include "include/dag_io.h"
#include "include/extended_std.h"
#include "include/layout_utils.h"
#include "include/pass_utils.h"
#include "include/program_context.h"
#include "include/t_layout_conversion_c.h"
#include "include/t_op.h"
#include "include/t_op_embrio.h"
#include "include/tensor_layout.h"
#include "include/utils.h"

namespace fhelipe {

namespace {

TensorLayout FillGaps(const TensorLayout& layout) {
  std::vector<TensorLayout::LayoutBit> result = layout.Bits();

  auto tensor_offset_bits = layout.TensorOffsetBits();
  if (result.empty()) {
    return {layout.GetShape(), result};
  }

  // Try to group continuous block together
  for (auto it = result.begin() + 1; it != result.end(); ++it) {
    auto& curr_bit = *it;
    const auto prev_bit = *(it - 1);
    if (prev_bit.has_value() && !curr_bit.has_value()) {
      const auto block_continuation = DimensionBit{
          prev_bit.value().dimension, prev_bit.value().bit_index + 1};
      if (Estd::contains(tensor_offset_bits, block_continuation)) {
        curr_bit = std::make_optional<DimensionBit>(block_continuation);
        int block_continuation_idx =
            Estd::find_index(tensor_offset_bits, block_continuation);
        tensor_offset_bits.erase(tensor_offset_bits.begin() +
                                 block_continuation_idx);
      }
    }
  }

  // Handle the rest
  auto curr_tensor_offset_bit = tensor_offset_bits.rbegin();
  for (auto it = result.rbegin(); it != result.rend(); ++it) {
    auto& bit = *it;
    if (curr_tensor_offset_bit == tensor_offset_bits.rend()) {
      break;
    }
    if (!bit.has_value()) {
      auto it2 = it;
      int num_gaps = std::count_if(
          it2, result.rend(), [](const auto& x) { return !x.has_value(); });
      while (num_gaps < tensor_offset_bits.rend() - curr_tensor_offset_bit) {
        ++curr_tensor_offset_bit;
      }

      bit = std::make_optional<DimensionBit>(*curr_tensor_offset_bit);
      ++curr_tensor_offset_bit;
    }
  }
  return {layout.GetShape(), result};
}

}  // namespace

std::vector<TensorLayout::LayoutBit> FillGapsLayoutPass::DefaultLayoutBits(
    const Shape& shape) {
  std::vector<TensorLayout::LayoutBit> layout_bits;
  for (int dim_idx : Estd::reverse(Estd::indices(shape.DimensionCount()))) {
    int dim_size = shape[dim_idx];
    for (int bit_idx = ceil_log2(dim_size) - 1; bit_idx > -1; --bit_idx) {
      layout_bits.emplace_back(DimensionBit(dim_idx, bit_idx));
    }
  }
  return Estd::reverse(layout_bits);
}

TensorLayout FillGapsLayoutPass::DefaultLayout(const Shape& shape,
                                               ChunkSize chunk_size) const {
  auto layout_bits = DefaultLayoutBits(shape);
  return {shape, ChunkBits(layout_bits, chunk_size)};
}

TensorLayout FillGapsLayoutPass::GetTStrideCOutputLayout(
    const TensorLayout& input_layout, const std::vector<Stride>& strides) {
  auto layout_bits = input_layout.Bits();
  Shape out_shape = GetOutputShapeTStrideC(input_layout.GetShape(), strides);
  for (auto& bit : layout_bits) {
    if (bit.has_value() &&
        (1 << bit.value().bit_index) < strides[bit.value().dimension].value()) {
      bit = std::nullopt;
    } else if (bit.has_value()) {
      bit.value().bit_index -= static_cast<int>(std::log2(
          static_cast<double>(strides.at(bit.value().dimension).value())));
    }
  }
  return {out_shape, layout_bits};
}

TensorLayout FillGapsLayoutPass::GetTResizeDimsCOutputLayout(
    const TensorLayout& input_layout, const Shape& output_shape) {
  auto layout_bits = input_layout.Bits();
  for (auto& bit : layout_bits) {
    if (bit.has_value() &&
        output_shape[bit.value().dimension] <= (1 << bit.value().bit_index)) {
      bit = std::nullopt;
    }
  }

  return FillGaps({output_shape, layout_bits});
}

TensorLayout FillGapsLayoutPass::GetTReduceDimCOutputLayout(
    const TensorLayout& input_layout, int dimension) {
  Shape new_shape =
      GetOutputShapeTReduceDimC(input_layout.GetShape(), dimension);
  new_shape[dimension] = 1;
  return GetTResizeDimsCOutputLayout(input_layout, new_shape);
}

TensorLayout FillGapsLayoutPass::GetTReplicateDimCOutputLayout(
    const TensorLayout& input_layout, int dimension, int multiple) {
  Shape new_shape = GetOutputShapeTReplicateDimsC(input_layout.GetShape(),
                                                  dimension, multiple);
  return GetTResizeDimsCOutputLayout(input_layout, new_shape);
}

TensorLayout FillGapsLayoutPass::GetTReorderDimsCOutputLayout(
    const TensorLayout& input_layout, const std::vector<int>& dim_order) {
  auto bits = input_layout.Bits();
  for (auto& bit : bits) {
    if (!bit.has_value()) {
      continue;
    }
    bit.value().dimension = Estd::find_index(dim_order, bit.value().dimension);
  }
  const auto& output_layout = TensorLayout(
      GetOutputShapeTReorderDimsC(input_layout.GetShape(), dim_order), bits);
  return output_layout;
}

TensorLayout FillGapsLayoutPass::StaticGetOutputLayout(
    const std::shared_ptr<Node<TOpEmbrio>>& embrio,
    const TensorLayout& input_layout) {
  if (const auto* t_reorder_dim_c =
          dynamic_cast<const TReorderDimsCEmbrio*>(&embrio->Value())) {
    return FillGapsLayoutPass::GetTReorderDimsCOutputLayout(
        input_layout, t_reorder_dim_c->DimensionOrder());
  }
  if (const auto* t_replicate_dim_c =
          dynamic_cast<const TReplicateDimCEmbrio*>(&embrio->Value())) {
    return FillGapsLayoutPass::GetTReplicateDimCOutputLayout(
        input_layout, t_replicate_dim_c->DimensionToReplicate(),
        t_replicate_dim_c->ReplicationMultiple());
  }
  if (const auto* t_reduce_dim_c =
          dynamic_cast<const TReduceDimCEmbrio*>(&embrio->Value())) {
    return FillGapsLayoutPass::GetTReduceDimCOutputLayout(
        input_layout, t_reduce_dim_c->DimensionToReduce());
  }
  if (const auto* t_stride_c =
          dynamic_cast<const TStrideCEmbrio*>(&embrio->Value())) {
    return FillGapsLayoutPass::GetTStrideCOutputLayout(input_layout,
                                                       t_stride_c->Strides());
  }
  if (const auto* t_stride_c =
          dynamic_cast<const TMergedStrideCEmbrio*>(&embrio->Value())) {
    return FillGapsLayoutPass::GetTStrideCOutputLayout(input_layout,
                                                       t_stride_c->Strides());
  }
  if (const auto* t_resize_dim_c =
          dynamic_cast<const TResizeDimCEmbrio*>(&embrio->Value())) {
    return FillGapsLayoutPass::GetTResizeDimsCOutputLayout(
        input_layout, embrio->Value().OutputShape());
  }
  return input_layout;
}

namespace {

std::vector<int> InversePermutation(const std::vector<int>& order) {
  std::vector<int> result(order.size());
  for (int i : Estd::indices(order.size())) {
    result[order[i]] = i;
  }
  return result;
}

}  // namespace

TensorLayout FillGapsLayoutPass::InverseStaticGetOutputLayout(
    const TOp& t_op, const TensorLayout& output_layout) {
  if (const auto* t_reorder_dim_c = dynamic_cast<const TReorderDimsC*>(&t_op)) {
    return FillGapsLayoutPass::GetTReorderDimsCOutputLayout(
        output_layout, InversePermutation(t_reorder_dim_c->DimensionOrder()));
  }
  if (const auto* t_replicate_dim_c =
          dynamic_cast<const TReplicateDimC*>(&t_op)) {
    return FillGapsLayoutPass::GetTReduceDimCOutputLayout(
        output_layout, t_replicate_dim_c->DimensionToReplicate());
  }
  if (const auto* t_reduce_dim_c = dynamic_cast<const TReduceDimC*>(&t_op)) {
    return FillGapsLayoutPass::GetTReplicateDimCOutputLayout(
        output_layout, t_reduce_dim_c->DimensionToReduce(),
        t_reduce_dim_c->InputLayout()
            .GetShape()[t_reduce_dim_c->DimensionToReduce()]);
  }
  if (const auto* t_stride_c = dynamic_cast<const TStrideC*>(&t_op)) {
    LOG(FATAL) << "Not implemented";
  }
  if (const auto* t_resize_dim_c = dynamic_cast<const TResizeDimC*>(&t_op)) {
    return FillGapsLayoutPass::GetTResizeDimsCOutputLayout(
        output_layout, t_resize_dim_c->InputLayout().GetShape());
  }
  if (const auto* t_drop_dim_c = dynamic_cast<const TDropDimC*>(&t_op)) {
    return TInsertDimC(output_layout, t_drop_dim_c->DimensionToDrop())
        .OutputLayout();
  }
  if (const auto* t_insert_dim_c = dynamic_cast<const TInsertDimC*>(&t_op)) {
    return TDropDimC(output_layout, t_insert_dim_c->DimensionToInsert())
        .OutputLayout();
  }
  return output_layout;
}

std::vector<std::shared_ptr<Node<TOp>>> FillGapsLayoutPass::MatchLayouts(
    Dag<TOp>& dag, const std::vector<std::shared_ptr<Node<TOp>>>& parents) {
  return MatchLayoutsForHoisting(dag, parents);
}

}  // namespace fhelipe
