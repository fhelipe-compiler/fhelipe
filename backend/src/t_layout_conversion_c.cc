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

#include "include/t_layout_conversion_c.h"

#include <glog/logging.h>

#include <string>

#include "include/ct_program.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_index.h"
#include "include/laid_out_tensor_utils.h"
#include "include/shape.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_generator.h"
#include "include/translation_mask_utils.h"

namespace fhelipe {

TOp::LaidOutTensorCt TLayoutConversionC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<LaidOutTensorCt>& input_tensors) const {
  const auto& input_tensor = input_tensors[0];
  CHECK(input_tensor.Layout() == input_layout_);
  const auto& input_layout = input_tensor.Layout();
  const auto& trans_masks =
      MakeTranslationMasks(input_layout, output_layout_,
                           [](const TensorIndex& src_ti) { return src_ti; });
  const auto& result = ApplyTranslationMasks(ct_program, input_tensor,
                                             trans_masks, output_layout_);
  return TOp::LaidOutTensorCt{AdaptToLayout(output_layout_, result)};
}

void TLayoutConversionC::SetLayouts(const TensorLayout& input_layout,
                                    const TensorLayout& output_layout) {
  input_layout_ = input_layout;
  output_layout_ = output_layout;
}

bool TLayoutConversionC::EqualTo(const TOp& other) const {
  const auto* t_layout_conversion_c =
      dynamic_cast<const TLayoutConversionC*>(&other);
  return t_layout_conversion_c &&
         OutputLayout() == t_layout_conversion_c->OutputLayout() &&
         InputLayout() == t_layout_conversion_c->InputLayout();
}

}  // namespace fhelipe
