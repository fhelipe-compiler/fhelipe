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

#include "include/t_insert_dim_c.h"

#include <glog/logging.h>

#include <algorithm>
#include <optional>

#include "include/array.h"
#include "include/ct_program.h"
#include "include/dimension_bit.h"

namespace fhelipe {

namespace {

std::vector<std::optional<DimensionBit>> GetOutputDimensionBits(
    std::vector<std::optional<DimensionBit>> bits, int dim_to_insert) {
  for (auto& bit : bits) {
    if (bit.has_value() && bit.value().dimension >= dim_to_insert) {
      ++bit.value().dimension;
    }
  }
  return bits;
}

TensorLayout GetOutputLayout(const TensorLayout& input_layout,
                             int dim_to_insert) {
  return {GetOutputShapeTInsertDimC(input_layout.GetShape(), dim_to_insert),
          GetOutputDimensionBits(input_layout.Bits(), dim_to_insert)};
}

void SanityCheckTInsertDimCLayouts(const TensorLayout& input_layout,
                                   const TensorLayout& output_layout,
                                   int dim_to_insert) {
  const Shape& shape = input_layout.GetShape();
  CHECK(shape.DimensionCount() >= dim_to_insert);
  CHECK(dim_to_insert >= 0);
  CHECK(output_layout == GetOutputLayout(input_layout, dim_to_insert));
}

}  // namespace

class CtOp;

TInsertDimC::TInsertDimC(const TensorLayout& layout, int dim_to_insert)
    : layout_(layout),
      output_layout_(GetOutputLayout(layout_, dim_to_insert)),
      dim_to_insert_(dim_to_insert) {
  SanityCheckTInsertDimCLayouts(layout_, output_layout_, dim_to_insert_);
}

void TInsertDimC::SetLayouts(const TensorLayout& input_layout,
                             const TensorLayout& output_layout) {
  SanityCheckTInsertDimCLayouts(input_layout, output_layout, dim_to_insert_);
  layout_ = input_layout;
  output_layout_ = output_layout;
}

Shape GetOutputShapeTInsertDimC(const Shape& shape, int dim_to_insert) {
  CHECK(dim_to_insert <= shape.DimensionCount());
  std::vector<int> new_shape(shape.begin(), shape.begin() + dim_to_insert);
  new_shape.push_back(1);
  new_shape.insert(new_shape.end(), shape.begin() + dim_to_insert, shape.end());
  return {Array(new_shape)};
}

const TensorLayout& TInsertDimC::OutputLayout() const { return output_layout_; }

TOp::LaidOutTensorCt TInsertDimC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  CHECK(input_tensors[0].Layout() == layout_);
  (void)ct_program;
  return TOp::LaidOutTensorCt{
      AdaptToLayout(OutputLayout(), input_tensors[0].Chunks())};
}

bool TInsertDimC::EqualTo(const TOp& other) const {
  const auto* t_insert_dim_c = dynamic_cast<const TInsertDimC*>(&other);
  return t_insert_dim_c && OutputLayout() == t_insert_dim_c->OutputLayout() &&
         DimensionToInsert() == t_insert_dim_c->DimensionToInsert();
}

}  // namespace fhelipe
