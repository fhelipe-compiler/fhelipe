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

#include "include/t_drop_dim_c.h"

#include <glog/logging.h>

#include <algorithm>
#include <optional>

#include "include/array.h"
#include "include/ct_program.h"
#include "include/dimension_bit.h"
#include "include/laid_out_tensor_utils.h"
#include "include/tensor_layout.h"

namespace fhelipe {

class CtOp;

Shape GetOutputShapeTDropDimC(const Shape& shape, int dim_to_drop) {
  CHECK(dim_to_drop < shape.DimensionCount());
  CHECK(shape[dim_to_drop] == 1);
  std::vector<int> new_shape(shape.begin(), shape.begin() + dim_to_drop);
  new_shape.insert(new_shape.end(), shape.begin() + dim_to_drop + 1,
                   shape.end());
  return {Array(new_shape)};
}

namespace {

std::vector<std::optional<DimensionBit>> GetOutputDimensionBits(
    std::vector<std::optional<DimensionBit>> bits, int dim_to_drop) {
  for (auto& bit : bits) {
    if (bit.has_value() && bit.value().dimension >= dim_to_drop) {
      --bit.value().dimension;
    }
  }
  return bits;
}

TensorLayout GetOutputLayout(const TensorLayout& input_layout,
                             int dim_to_drop) {
  return {GetOutputShapeTDropDimC(input_layout.GetShape(), dim_to_drop),
          GetOutputDimensionBits(input_layout.Bits(), dim_to_drop)};
}

void SanityCheckTDropDimCLayouts(const TensorLayout& input_layout,
                                 const TensorLayout& output_layout,
                                 int dim_to_drop) {
  Shape shape = input_layout.GetShape();
  CHECK(shape.DimensionCount() > dim_to_drop);
  CHECK(shape[dim_to_drop] == 1);
  CHECK(output_layout == GetOutputLayout(input_layout, dim_to_drop));
}

}  // namespace

TDropDimC::TDropDimC(const TensorLayout& layout, int dim_to_drop)
    : layout_(layout),
      output_layout_(GetOutputLayout(layout, dim_to_drop)),
      dim_to_drop_(dim_to_drop) {
  SanityCheckTDropDimCLayouts(layout_, output_layout_, dim_to_drop_);
}

void TDropDimC::SetLayouts(const TensorLayout& input_layout,
                           const TensorLayout& output_layout) {
  SanityCheckTDropDimCLayouts(input_layout, output_layout, dim_to_drop_);
  layout_ = input_layout;
  output_layout_ = output_layout;
}

TOp::LaidOutTensorCt TDropDimC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  CHECK(input_tensors[0].Layout() == layout_);
  (void)ct_program;
  return TOp::LaidOutTensorCt{
      AdaptToLayout(OutputLayout(), input_tensors[0].Chunks())};
}

bool TDropDimC::EqualTo(const TOp& other) const {
  const auto* t_drop_dim_c = dynamic_cast<const TDropDimC*>(&other);
  return t_drop_dim_c && t_drop_dim_c->OutputLayout() == OutputLayout() &&
         DimensionToDrop() == t_drop_dim_c->DimensionToDrop();
}

}  // namespace fhelipe
