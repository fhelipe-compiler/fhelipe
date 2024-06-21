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

#ifndef FHELIPE_RAW_SHIFT_BIT_H_
#define FHELIPE_RAW_SHIFT_BIT_H_

#include <glog/logging.h>
#include <stdlib.h>

#include "dimension_bit.h"
#include "include/shape.h"
#include "tensor_index.h"
#include "tensor_layout.h"

namespace fhelipe {

class RawShiftBit {
 public:
  RawShiftBit(const DimensionBit& dim_bit, int direction)
      : dim_bit_(dim_bit), direction_(direction) {
    CHECK(std::abs(direction) == 1);
  }

  int Dimension() const { return dim_bit_.dimension; }
  int Direction() const { return direction_; }
  DimensionBit GetDimensionBit() const { return dim_bit_; }
  DiffTensorIndex ShiftDiff(const Shape& shape) const;
  int ShiftAmount() const { return direction_ * (1 << dim_bit_.bit_index); }

 private:
  DimensionBit dim_bit_;
  int direction_;
};

}  // namespace fhelipe

#endif  // FHELIPE_RAW_SHIFT_BIT_H_
