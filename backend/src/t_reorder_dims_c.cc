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

#include "include/t_reorder_dims_c.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <utility>

#include "include/array.h"
#include "include/dimension_bit.h"
#include "include/extended_std.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_utils.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

Shape GetOutputShapeTReorderDimsC(const Shape& shape,
                                  const std::vector<int>& dim_order) {
  return {Array(Estd::permute(shape, dim_order))};
}

TOp::LaidOutTensorCt TReorderDimsC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  const auto& input_tensor = input_tensors[0];
  CHECK(input_tensor.Layout() == input_layout_);
  auto output_layout = OutputLayout();
  const std::vector<TranslationMask>& translation_masks = MakeTranslationMasks(
      input_layout_, output_layout,
      [&output_layout, this](const TensorIndex& ti) {
        return TensorIndex(output_layout.GetShape(),
                           Estd::permute(ti.DimensionIndices(), dim_order_));
      });
  const auto& result = ApplyTranslationMasks(ct_program, input_tensors[0],
                                             translation_masks, output_layout);
  return TOp::LaidOutTensorCt{result};
}

void SanityCheckTReorderDimsCLayouts(const TensorLayout& input_layout,
                                     const TensorLayout& output_layout,
                                     std::vector<int> dim_order) {
  CHECK(output_layout.GetShape() ==
        GetOutputShapeTReorderDimsC(input_layout.GetShape(), dim_order));
  std::sort(dim_order.begin(), dim_order.end());
  CHECK(dim_order == Estd::indices(input_layout.GetShape().DimensionCount()));
}

TReorderDimsC::TReorderDimsC(const TensorLayout& input_layout,
                             const TensorLayout& output_layout,
                             const std::vector<int>& dim_order)
    : input_layout_(input_layout),
      output_layout_(output_layout),
      dim_order_(dim_order) {
  SanityCheckTReorderDimsCLayouts(input_layout, output_layout, dim_order);
}

void TReorderDimsC::SetLayouts(const TensorLayout& input_layout,
                               const TensorLayout& output_layout) {
  SanityCheckTReorderDimsCLayouts(input_layout, output_layout, dim_order_);
  input_layout_ = input_layout;
  output_layout_ = output_layout;
}

bool TReorderDimsC::EqualTo(const TOp& other) const {
  const auto* t_reorder_dims_c = dynamic_cast<const TReorderDimsC*>(&other);
  return t_reorder_dims_c &&
         t_reorder_dims_c->OutputLayout() == OutputLayout() &&
         t_reorder_dims_c->InputLayout() == InputLayout() &&
         t_reorder_dims_c->DimensionOrder() == DimensionOrder();
}

}  // namespace fhelipe
