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
#include "include/t_reduce_dim_c.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "test/test_constants.h"
#include "test/test_utils.h"

using namespace fhelipe;

int dim_to_reduce = -1;

Tensor<std::optional<PtVal>> CreateReduced(const Tensor<PtVal>& tensor,
                                           int dim_to_reduce) {
  Shape in_shape = tensor.GetShape();
  Shape out_shape = in_shape;
  out_shape[dim_to_reduce] = 1;
  std::vector<PtVal> values(out_shape.ValueCnt());
  for (int flat_idx : Estd::indices(out_shape.ValueCnt())) {
    auto ti = TensorIndex(out_shape, flat_idx);
    auto indices = ti.DimensionIndices();
    PtVal reduction = 0;
    for (int idx : Estd::indices(in_shape[dim_to_reduce])) {
      indices[dim_to_reduce] = idx;
      reduction += tensor[TensorIndex(in_shape, indices)];
    }
    values[ti.Flat()] = reduction;
  }
  return ToOptionalTensor({out_shape, values});
}

RamDictionary<Tensor<std::optional<PtVal>>> CreateTReduceDimCCheck(
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  RamDictionary<Tensor<std::optional<PtVal>>> result;
  result.Record("out0", CreateReduced(tensor_dict.At("in0"), dim_to_reduce));
  return result;
}

Dag<TOp> CreateTReduceDimCTOpDag() {
  Dag<TOp> top_dag;
  auto input_layout = RandomLayout();
  dim_to_reduce = rand() % input_layout.GetShape().DimensionCount();
  const auto& a = MakeInputNode(top_dag, input_layout, "in0");
  const auto& node =
      top_dag.AddNode(std::make_unique<TReduceDimC>(
                          input_layout,
                          FillGapsLayoutPass::GetTReduceDimCOutputLayout(
                              input_layout, dim_to_reduce),
                          dim_to_reduce),
                      {a});
  MakeOutputNode(top_dag, node, "out0");
  return top_dag;
}

TEST(TReduceDimCTest, Basic) {
  DoTest<Cleartext>(CreateTReduceDimCTOpDag, CreateTReduceDimCCheck);
}
