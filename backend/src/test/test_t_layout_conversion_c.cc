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
#include "include/t_layout_conversion_c.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "test/test_constants.h"
#include "test/test_utils.h"

using namespace fhelipe;

RamDictionary<Tensor<std::optional<PtVal>>> CreateTLayoutConversionCCheck(
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  RamDictionary<Tensor<std::optional<PtVal>>> result;
  result.Record("out0", ToOptionalTensor(tensor_dict.At("in0")));
  return result;
}

Dag<TOp> CreateTLayoutConversionCTOpDag() {
  Dag<TOp> top_dag;
  auto input_layout = RandomLayout();
  auto output_layout =
      RandomLayout(input_layout.GetShape(), input_layout.ChunkSize());
  const auto& a = MakeInputNode(top_dag, input_layout, "in0");
  const auto& node = top_dag.AddNode(
      std::make_unique<TLayoutConversionC>(input_layout, output_layout), {a});
  MakeOutputNode(top_dag, node, "out0");
  return top_dag;
}

TEST(TLayoutConversionCTest, Basic) {
  DoTest<Cleartext>(CreateTLayoutConversionCTOpDag,
                    CreateTLayoutConversionCCheck);
}
