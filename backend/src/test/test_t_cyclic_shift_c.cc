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
#include "include/t_cyclic_shift_c.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "test/test_constants.h"
#include "test/test_utils.h"

using namespace fhelipe;

namespace {

DiffTensorIndex shift_by = RandomDiffTensorIndex(RandomShape());

Tensor<std::optional<PtVal>> ComputeShifted(const Tensor<PtVal>& tensor,
                                            const DiffTensorIndex& shift_by) {
  CHECK(tensor.GetShape() == shift_by.GetShape());
  Shape shape = tensor.GetShape();
  Tensor<PtVal> result(tensor.GetShape(), std::vector<PtVal>(shape.ValueCnt()));
  for (int flat_idx : Estd::indices(shape.ValueCnt())) {
    result[shift_by.CyclicAdd(TensorIndex(shape, flat_idx))] =
        tensor[TensorIndex(shape, flat_idx)];
  }
  return ToOptionalTensor(result);
}

}  // namespace

RamDictionary<Tensor<std::optional<PtVal>>> CreateCyclicShiftCCheck(
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  RamDictionary<Tensor<std::optional<PtVal>>> result;
  result.Record("out0", ComputeShifted(tensor_dict.At("in0"), shift_by));
  return result;
}

Dag<TOp> CreateCyclicShiftCTOpDag() {
  auto input_layout = RandomLayout();
  Dag<TOp> top_dag;
  const auto& a = MakeInputNode(top_dag, input_layout, "in0");
  shift_by = RandomDiffTensorIndex(input_layout.GetShape());
  const auto& node = top_dag.AddNode(
      std::make_unique<TCyclicShiftC>(input_layout, shift_by), {a});
  MakeOutputNode(top_dag, node, "out0");
  return top_dag;
}

TEST(TCyclicShiftCTest, Basic) {
  DoTest<Cleartext>(CreateCyclicShiftCTOpDag, CreateCyclicShiftCCheck);
}
