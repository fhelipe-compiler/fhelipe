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

#include "include/t_cyclic_shift_c.h"

#include <glog/logging.h>

#include <string>
#include <vector>

#include "include/ct_program.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_index.h"
#include "include/shape.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_generator.h"
#include "include/translation_mask_utils.h"

namespace fhelipe {

TOp::LaidOutTensorCt TCyclicShiftC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  CHECK(input_tensors[0].Layout() == layout_);
  const auto& input_layout = input_tensors[0].Layout();
  const std::vector<TranslationMask>& trans_masks = MakeTranslationMasks(
      input_layout, input_layout, [this](const TensorIndex& src_ti) {
        return (this->rotate_by_).CyclicAdd(src_ti);
      });
  const auto& result = ApplyTranslationMasks(ct_program, input_tensors[0],
                                             trans_masks, input_layout);
  return TOp::LaidOutTensorCt{result};
}

void TCyclicShiftC::SetLayouts(const TensorLayout& input_layout,
                               const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  layout_ = input_layout;
}

bool TCyclicShiftC::EqualTo(const TOp& other) const {
  const auto* t_cyclic_shift_c = dynamic_cast<const TCyclicShiftC*>(&other);
  return t_cyclic_shift_c &&
         t_cyclic_shift_c->OutputLayout() == OutputLayout() &&
         GetDiffTensorIndex() == t_cyclic_shift_c->GetDiffTensorIndex();
}

}  // namespace fhelipe
