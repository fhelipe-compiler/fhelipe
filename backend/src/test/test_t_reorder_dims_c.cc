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
#include "include/t_reorder_dims_c.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "test/test_constants.h"
#include "test/test_utils.h"

using namespace fhelipe;

std::vector<int> dim_order;

Tensor<std::optional<PtVal>> CreateReordered(
    const Tensor<PtVal>& tensor, const std::vector<int>& dim_order) {
  Shape in_shape = tensor.GetShape();
  Shape out_shape(fhelipe::Array(Estd::permute(
      std::vector<int>(in_shape.begin(), in_shape.end()), dim_order)));
  std::vector<PtVal> values(out_shape.ValueCnt());
  for (int flat_idx : Estd::indices(in_shape.ValueCnt())) {
    const auto& ti = TensorIndex(in_shape, flat_idx);
    Array dim_indices = ti.DimensionIndices();
    auto new_dim_indices = fhelipe::Array(Estd::permute(
        std::vector<int>(dim_indices.begin(), dim_indices.end()), dim_order));
    const auto& new_ti = TensorIndex(out_shape, new_dim_indices);
    values[new_ti.Flat()] = tensor[ti];
  }
  return ToOptionalTensor({out_shape, values});
}

RamDictionary<Tensor<std::optional<PtVal>>> CreateTReorderDimsCCheck(
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  RamDictionary<Tensor<std::optional<PtVal>>> result;
  result.Record("out0", CreateReordered(tensor_dict.At("in0"), dim_order));
  return result;
}

Dag<TOp> CreateTReorderDimsCTOpDag() {
  Dag<TOp> top_dag;
  auto input_layout = RandomLayout();
  dim_order =
      Estd::shuffle(Estd::indices(input_layout.GetShape().DimensionCount()));
  const auto& a = MakeInputNode(top_dag, input_layout, "in0");
  const auto& node =
      top_dag.AddNode(std::make_unique<TReorderDimsC>(
                          input_layout,
                          FillGapsLayoutPass::GetTReorderDimsCOutputLayout(
                              input_layout, dim_order),
                          dim_order),
                      {a});
  MakeOutputNode(top_dag, node, "out0");
  return top_dag;
}

TEST(TReorderDimsCTest, Basic) {
  DoTest<Cleartext>(CreateTReorderDimsCTOpDag, &CreateTReorderDimsCCheck);
}
