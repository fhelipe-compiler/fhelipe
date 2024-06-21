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

#include "include/t_add_cc.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <vector>

#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/laid_out_tensor.h"
#include "include/tensor_layout.h"

namespace fhelipe {
class CtOp;

TAddCC::TAddCC(const TensorLayout& layout) : layout_(layout) {}

TOp::LaidOutTensorCt TAddCC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 2);
  CHECK(input_tensors[0].Layout() == layout_);
  CHECK(input_tensors[1].Layout() == layout_);
  auto result = Estd::transform(
      input_tensors[0].Chunks(), input_tensors[1].Chunks(),
      [&ct_program](const auto& lhs, const auto& rhs) {
        CHECK(lhs.Offset() == rhs.Offset());
        return TOp::LaidOutChunk(
            lhs.Layout(), lhs.Offset(),
            ct_program::CreateAddCC(ct_program, lhs.Chunk(), rhs.Chunk()));
      });
  return TOp::LaidOutTensorCt{result};
}

void TAddCC::SetLayouts(const TensorLayout& input_layout,
                        const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
}

bool TAddCC::EqualTo(const TOp& other) const {
  const auto* t_add_cc = dynamic_cast<const TAddCC*>(&other);
  return t_add_cc && t_add_cc->OutputLayout() == OutputLayout();
}

}  // namespace fhelipe
