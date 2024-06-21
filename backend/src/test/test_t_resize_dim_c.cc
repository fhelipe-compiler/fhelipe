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
#include "include/t_resize_dim_c.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "test/test_constants.h"
#include "test/test_utils.h"

using namespace fhelipe;

Shape output_shape = RandomShape();

Tensor<std::optional<PtVal>> ComputeResized(const Tensor<PtVal>& tensor,
                                            const Shape& output_shape) {
  std::vector<PtVal> values(output_shape.ValueCnt());
  for (int flat_idx : Estd::indices(output_shape.ValueCnt())) {
    const auto& output_index = TensorIndex(output_shape, flat_idx);
    const auto& dim_indices = output_index.DimensionIndices();
    if (IsInRange(output_shape, dim_indices) &&
        IsInRange(tensor.GetShape(), dim_indices)) {
      values[output_index.Flat()] =
          tensor[TensorIndex(tensor.GetShape(), dim_indices)];
    } else if (IsInRange(output_shape, dim_indices)) {
      values[output_index.Flat()] = 0;
    }
  }
  return ToOptionalTensor({output_shape, values});
}

RamDictionary<Tensor<std::optional<PtVal>>> CreateTResizeDimCCheck(
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  RamDictionary<Tensor<std::optional<PtVal>>> result;
  result.Record("out0", ComputeResized(tensor_dict.At("in0"), output_shape));
  return result;
}

Dag<TOp> CreateTResizeDimCTOpDag() {
  auto input_layout = RandomLayout();
  output_shape = RandomShape(input_layout.GetShape().DimensionCount());
  Dag<TOp> top_dag;
  const auto& a = MakeInputNode(top_dag, input_layout, "in0");
  const auto& node = top_dag.AddNode(
      std::make_unique<TResizeDimC>(
          input_layout, FillGapsLayoutPass::GetTResizeDimsCOutputLayout(
                            input_layout, output_shape)),
      {a});
  MakeOutputNode(top_dag, node, "out0");
  return top_dag;
}

TEST(TResizeDimCTest, Basic) {
  DoTest<Cleartext>(CreateTResizeDimCTOpDag, CreateTResizeDimCCheck);
}
