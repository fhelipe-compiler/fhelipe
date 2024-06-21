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

#include "include/t_mul_csi.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>

#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/laid_out_tensor.h"
#include "include/tensor_layout.h"

namespace fhelipe {
class CtOp;

TOp::LaidOutTensorCt TMulCSI::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  CHECK(input_tensors[0].Layout() == layout_);
  auto result =
      Estd::transform(input_tensors[0].Chunks(), [&](const auto& lhs) {
        return TOp::LaidOutChunk{
            lhs.Layout(), lhs.Offset(),
            ct_program::CreateMulCS(ct_program, lhs.Chunk(), scalar_)};
      });
  return TOp::LaidOutTensorCt{result};
}

void TMulCSI::SetLayouts(const TensorLayout& input_layout,
                         const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
}

bool TMulCSI::EqualTo(const TOp& other) const {
  const auto* t_mul_csi = dynamic_cast<const TMulCSI*>(&other);
  return t_mul_csi && t_mul_csi->OutputLayout() == OutputLayout() &&
         Scalar() == t_mul_csi->Scalar();
}

}  // namespace fhelipe
