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

#include "include/shape.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "include/array.h"

namespace fhelipe {

int Shape::ValueCnt() const {
  return std::accumulate(begin(), end(), 1, std::multiplies<>());
}

Shape Shape::SubShape(int start_i, int end_i) const {
  CHECK(start_i >= 0);
  CHECK(end_i >= 0);
  CHECK(start_i < DimensionCount());
  CHECK(end_i < DimensionCount());
  return Shape(begin() + start_i, begin() + end_i);
}

Shape ShapeWithChannels(int channel_cnt, const Shape& spatial_shape) {
  std::vector<int> result({channel_cnt});
  for (int spatial_dim : spatial_shape) {
    result.push_back(spatial_dim);
  }
  return Shape(Array(result));
}

bool IsInRange(const Shape& shape, const Array& indices) {
  CHECK(shape.DimensionCount() == indices.size());
  for (int dim = 0; dim < indices.size(); ++dim) {
    if (shape[dim] <= indices[dim] || indices[dim] < 0) {
      return false;
    }
  }
  return true;
}

void Shape::CheckDims() const {
  for (int dim : dimensions_) {
    CHECK(dim >= 1);
  }
}

}  // namespace fhelipe
