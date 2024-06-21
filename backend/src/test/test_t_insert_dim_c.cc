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
#include "include/io_manager.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_dictionary.h"
#include "include/packer.h"
#include "include/persisted_dictionary.h"
#include "include/plaintext.h"
#include "include/plaintext_chunk.h"
#include "include/ram_dictionary.h"
#include "include/shape.h"
#include "include/t_insert_dim_c.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "test/test_constants.h"
#include "test/test_utils.h"

using namespace fhelipe;

int dim_to_insert = -1;

Tensor<std::optional<PtVal>> ComputeInserted(const Tensor<PtVal>& tensor,
                                             int dim_to_insert) {
  Shape in_shape = tensor.GetShape();
  std::vector<int> out_array(in_shape.begin(),
                             in_shape.begin() + dim_to_insert);
  out_array.push_back(1);
  Estd::append(out_array, std::vector<int>(in_shape.begin() + dim_to_insert,
                                           in_shape.end()));
  Shape new_shape((Array(out_array)));
  std::vector<PtVal> values(new_shape.ValueCnt());
  for (int flat_idx : Estd::indices(in_shape.ValueCnt())) {
    const auto& ti = TensorIndex(in_shape, flat_idx);
    const auto& new_ti = TensorIndex(new_shape, flat_idx);
    values[new_ti.Flat()] = tensor[ti];
  }
  return ToOptionalTensor({new_shape, values});
}

RamDictionary<Tensor<std::optional<PtVal>>> CreateTInsertDimCCheck(
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  RamDictionary<Tensor<std::optional<PtVal>>> result;
  result.Record("out0", ComputeInserted(tensor_dict.At("in0"), dim_to_insert));
  return result;
}

Dag<TOp> CreateTInsertDimCTOpDag() {
  auto input_layout = RandomLayout();
  dim_to_insert = rand() % (1 + input_layout.GetShape().DimensionCount());
  Dag<TOp> top_dag;
  const auto& a = MakeInputNode(top_dag, input_layout, "in0");
  const auto& node = top_dag.AddNode(
      std::make_unique<TInsertDimC>(input_layout, dim_to_insert), {a});
  MakeOutputNode(top_dag, node, "out0");
  return top_dag;
}

TEST(TInsertDimCTest, Basic) {
  DoTest<Cleartext>(CreateTInsertDimCTOpDag, CreateTInsertDimCCheck);
}
