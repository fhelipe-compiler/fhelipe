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

#include "include/cleartext.h"
#include "include/constants.h"
#include "include/dictionary.h"
#include "include/evaluator.h"
#include "include/fill_gaps_layout_pass.h"
#include "include/io_manager.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_dictionary.h"
#include "include/packer.h"
#include "include/persisted_dictionary.h"
#include "include/plaintext.h"
#include "include/plaintext_chunk.h"
#include "include/ram_dictionary.h"
#include "include/shape.h"
#include "include/t_stride_c.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "test/test_constants.h"
#include "test/test_utils.h"

using namespace fhelipe;

std::vector<Stride> strides;

Shape StridedShape(const Shape& in_shape, const std::vector<Stride>& strides) {
  std::vector<int> dimension_sizes;
  for (int dim : Estd::indices(in_shape.DimensionCount())) {
    dimension_sizes.push_back(
        std::ceil(in_shape[dim] / static_cast<double>(strides[dim].value())));
  }
  return {Array(dimension_sizes)};
}

std::vector<Stride> RandomStrides(const Shape& shape) {
  std::vector<Stride> result;
  result.reserve(shape.DimensionCount());
  for (int dim : Estd::indices(shape.DimensionCount())) {
    (void)dim;
    result.emplace_back(Stride(1 << (std::rand() % kMaxLogTestStrideSize)));
  }
  return result;
}

TensorIndex StridedIndex(const Shape& in_shape, const TensorIndex& ti,
                         const std::vector<Stride>& strides) {
  std::vector<int> result;
  for (int dim_idx : Estd::indices(ti.GetShape().DimensionCount())) {
    result.push_back(ti[dim_idx] * strides[dim_idx].value());
  }
  return {in_shape, result};
}

Tensor<std::optional<PtVal>> ComputeStrided(
    const Tensor<PtVal>& tensor, const std::vector<Stride>& strides) {
  Shape in_shape = tensor.GetShape();
  auto out_shape = StridedShape(in_shape, strides);
  std::vector<PtVal> values(out_shape.ValueCnt());
  for (int flat_idx : Estd::indices(values.size())) {
    values[flat_idx] = tensor[StridedIndex(
        in_shape, TensorIndex(out_shape, flat_idx), strides)];
  }
  return ToOptionalTensor({out_shape, values});
}

RamDictionary<Tensor<std::optional<PtVal>>> CreateTStrideCCheck(
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  RamDictionary<Tensor<std::optional<PtVal>>> result;
  result.Record("out0", ComputeStrided(tensor_dict.At("in0"), strides));
  return result;
}

Dag<TOp> CreateTStrideCTOpDag() {
  auto input_layout = RandomLayout();
  strides = RandomStrides(input_layout.GetShape());
  Dag<TOp> top_dag;
  const auto& a = MakeInputNode(top_dag, input_layout, "in0");
  const auto& node = top_dag.AddNode(
      std::make_unique<TStrideC>(
          input_layout,
          FillGapsLayoutPass::GetTStrideCOutputLayout(input_layout, strides),
          strides),
      {a});
  MakeOutputNode(top_dag, node, "out0");
  return top_dag;
}

TEST(TStrideCTest, Basic) {
  DoTest<Cleartext>(CreateTStrideCTOpDag, CreateTStrideCCheck);
}
