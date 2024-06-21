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

#include "include/t_add_cp.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>

#include "include/ct_program.h"
#include "include/laid_out_tensor.h"
#include "include/maybe_tensor_index.h"
#include "include/t_op_utils.h"
#include "include/tensor_layout.h"

namespace fhelipe {

class CtOp;

TOp::LaidOutTensorCt TAddCP::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  CHECK(input_tensors[0].Layout() == layout_);
  return CreateCtPtTensorOp(ct_program, input_tensors[0], pt_tensor_name_,
                            pt_tensor_log_scale_, &ct_program::CreateAddCP);
}

void TAddCP::SetLayouts(const TensorLayout& input_layout,
                        const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
}

bool TAddCP::EqualTo(const TOp& other) const {
  const auto* t_add_cp = dynamic_cast<const TAddCP*>(&other);
  return t_add_cp && other.OutputLayout() == OutputLayout() &&
         pt_tensor_log_scale_ == t_add_cp->PtTensorLogScale() &&
         pt_tensor_name_ == t_add_cp->PtTensorName();
}

}  // namespace fhelipe
