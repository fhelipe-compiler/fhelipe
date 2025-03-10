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
#include "include/evaluator.h"
#include "include/io_manager.h"
#include "include/t_input_c.h"
#include "include/t_mul_cp.h"
#include "include/t_output_c.h"
#include "include/tensor_layout.h"
#include "test/test_utils.h"

using namespace fhelipe;

RamDictionary<Tensor<std::optional<PtVal>>> CreateMulCPCheck(
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  RamDictionary<Tensor<std::optional<PtVal>>> result;
  auto inputs = GetValues(tensor_dict);
  result.Record(
      "out0",
      ToOptionalTensor({inputs[0].GetShape(),
                        Estd::transform(inputs[0].Values(), inputs[1].Values(),
                                        std::multiplies<>())}));
  return result;
}

Dag<TOp> CreateMulCPTOpDag() {
  auto input_layout = RandomLayout();
  Dag<TOp> top_dag;
  const auto& a = MakeInputNode(top_dag, input_layout, "in0");
  const auto& mul_cp = top_dag.AddNode(
      std::make_unique<TMulCP>(input_layout, "tensor", LogScale(50)), {a});
  MakeOutputNode(top_dag, mul_cp, "out0");
  return top_dag;
}

TEST(TMulCPTest, Basic) {
  DoTest<Cleartext>(CreateMulCPTOpDag, CreateMulCPCheck);
}
