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

#include "include/t_stride_c.h"

#include <algorithm>
#include <cmath>
#include <cwchar>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "include/array.h"
#include "include/ct_program.h"
#include "include/dimension_bit.h"
#include "include/extended_std.h"
#include "include/index_mask.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_utils.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_utils.h"

namespace fhelipe {
class CtOp;

namespace {

bool KeepIndexAfterStride(const std::vector<Stride>& strides,
                          const Array& dim_indices) {
  CHECK(strides.size() == dim_indices.size());
  for (int dim = 0; dim < strides.size(); ++dim) {
    if (dim_indices[dim] % strides[dim].value() != 0) {
      return false;
    }
  }
  return true;
}

Array DimensionIndicesAfterStride(const std::vector<Stride>& strides,
                                  Array dim_indices) {
  for (int idx : Estd::indices(dim_indices.size())) {
    dim_indices[idx] /= strides[idx].value();
  }
  return dim_indices;
}

}  // namespace

Shape GetOutputShapeTStrideC(const Shape& shape,
                             const std::vector<Stride>& strides) {
  CHECK(shape.DimensionCount() == strides.size());
  std::vector<int> dim_sizes = Estd::transform(
      std::vector<int>(shape.begin(), shape.end()), strides,
      [](int dim, const Stride& stride) {
        return static_cast<int>(
            std::ceil(dim / static_cast<double>(stride.value())));
      });
  return {Array(dim_sizes)};
}

void SanityCheckTStrideC(const TensorLayout& input_layout,
                         const TensorLayout& output_layout,
                         const std::vector<Stride>& strides) {
  for (int idx : Estd::indices(input_layout.GetShape().DimensionCount())) {
    int should_be =
        static_cast<int>(std::ceil(input_layout.GetShape()[idx] /
                                   static_cast<double>(strides[idx].value())));
    CHECK(should_be == output_layout.GetShape()[idx]);
  }
  CHECK(input_layout.GetShape().DimensionCount() ==
        output_layout.GetShape().DimensionCount());
}

TStrideC::TStrideC(const TensorLayout& input_layout,
                   const TensorLayout& output_layout,
                   const std::vector<Stride>& strides)
    : input_layout_(input_layout),
      output_layout_(output_layout),
      strides_(strides.begin(), strides.end()) {
  SanityCheckTStrideC(input_layout, output_layout, strides);
}

void TStrideC::SetLayouts(const TensorLayout& input_layout,
                          const TensorLayout& output_layout) {
  SanityCheckTStrideC(input_layout, output_layout, strides_);
  input_layout_ = input_layout;
  output_layout_ = output_layout;
}

TOp::LaidOutTensorCt TStrideC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  const auto& input_tensor = input_tensors[0];
  CHECK(input_tensor.Layout() == input_layout_);
  auto output_layout = OutputLayout();
  const std::vector<TranslationMask>& translation_masks = MakeTranslationMasks(
      input_layout_, output_layout,
      [this, &output_layout](const TensorIndex& ti) {
        const auto& dim_indices = ti.DimensionIndices();
        return KeepIndexAfterStride(strides_, dim_indices)
                   ? std::make_optional<TensorIndex>(
                         output_layout.GetShape(),
                         DimensionIndicesAfterStride(strides_, dim_indices))
                   : std::nullopt;
      });
  const auto& result = ApplyTranslationMasks(ct_program, input_tensors[0],
                                             translation_masks, output_layout);
  return TOp::LaidOutTensorCt{result};
}

bool TStrideC::EqualTo(const TOp& other) const {
  const auto* t_stride_c = dynamic_cast<const TStrideC*>(&other);
  return t_stride_c && t_stride_c->OutputLayout() == OutputLayout() &&
         t_stride_c->InputLayout() == InputLayout() &&
         Estd::transform(t_stride_c->Strides(), [](auto x) {
           return x.value();
         }) == Estd::transform(Strides(), [](auto x) { return x.value(); });
}

}  // namespace fhelipe
