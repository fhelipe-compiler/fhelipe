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

#include "include/t_resize_dim_c.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "include/array.h"
#include "include/chunk_ir.h"
#include "include/ct_program.h"
#include "include/dimension_bit.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_index.h"
#include "include/laid_out_tensor_utils.h"
#include "include/shape.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_utils.h"
#include "include/utils.h"

namespace fhelipe {
class CtOp;

void SanityCheckTResizeDimC(const TensorLayout& input_layout,
                            const TensorLayout& output_layout) {
  CHECK(input_layout.GetShape().DimensionCount() ==
        output_layout.GetShape().DimensionCount());
}

TResizeDimC::TResizeDimC(const TensorLayout& input_layout,
                         const TensorLayout& output_layout)
    : input_layout_(input_layout), output_layout_(output_layout) {
  SanityCheckTResizeDimC(input_layout, output_layout);
}

void TResizeDimC::SetLayouts(const TensorLayout& input_layout,
                             const TensorLayout& output_layout) {
  SanityCheckTResizeDimC(input_layout, output_layout);
  input_layout_ = input_layout;
  output_layout_ = output_layout;
}

TOp::LaidOutTensorCt TResizeDimC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  const auto& input_tensor = input_tensors[0];
  CHECK(input_tensor.Layout() == input_layout_);
  if (input_layout_ == output_layout_) {
    return input_tensors[0];
  }

  const std::vector<TranslationMask>& translation_masks = MakeTranslationMasks(
      input_layout_, output_layout_, [this](const TensorIndex& ti) {
        return IsInRange(output_layout_.GetShape(), ti.DimensionIndices())
                   ? std::make_optional<TensorIndex>(TensorIndex(
                         output_layout_.GetShape(), ti.DimensionIndices()))
                   : std::nullopt;
      });
  const auto& result = ApplyTranslationMasks(ct_program, input_tensors[0],
                                             translation_masks, output_layout_);
  return TOp::LaidOutTensorCt{result};
}

bool TResizeDimC::EqualTo(const TOp& other) const {
  const auto* t_resize_dim_c = dynamic_cast<const TResizeDimC*>(&other);
  return t_resize_dim_c && t_resize_dim_c->OutputLayout() == OutputLayout() &&
         t_resize_dim_c->InputLayout() == InputLayout();
}

int TResizeDimC::BackendMaskDepth() const {
  return input_layout_ != output_layout_;
}

}  // namespace fhelipe
