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

#include "include/raw_shift_bit.h"

#include <vector>

#include "include/array.h"
#include "include/dimension_bit.h"
#include "include/tensor_index.h"

namespace fhelipe {

DiffTensorIndex RawShiftBit::ShiftDiff(const Shape& shape) const {
  std::vector<int> shift_indices(shape.DimensionCount());
  shift_indices[dim_bit_.dimension] = direction_ * (1 << dim_bit_.bit_index);
  return DiffTensorIndex(shape, Array(shift_indices));
}

}  // namespace fhelipe
