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

#include "include/tensor_index.h"

#include <glog/logging.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

#include "include/array.h"
#include "include/shape.h"

namespace fhelipe {

namespace {

int MakeFlatIndex(const Shape& shape, const Array& dimension_indices) {
  CHECK(IsInRange(shape, dimension_indices));
  int flat_index = 0;
  for (int i = 0; i < dimension_indices.size(); ++i) {
    flat_index *= shape[i];
    flat_index += dimension_indices[i];
  }
  return flat_index;
}

}  // namespace

TensorIndex::TensorIndex(const Shape& shape, int flat)
    : shape_(shape),
      flat_index_(flat),
      dimension_indices_(shape.DimensionCount()) {
  for (int i = shape.DimensionCount() - 1; i >= 0; --i) {
    dimension_indices_[i] = flat % shape_[i];
    flat /= shape_[i];
  }
}

TensorIndex::TensorIndex(const Shape& shape,
                         const std::vector<int>& dimension_indices)
    : shape_(shape),
      flat_index_(MakeFlatIndex(
          shape, Array(dimension_indices.begin(), dimension_indices.end()))),
      dimension_indices_(dimension_indices.begin(), dimension_indices.end()) {}

TensorIndex::TensorIndex(const Shape& shape, const Array& dimension_indices)
    : shape_(shape),
      flat_index_(MakeFlatIndex(shape, dimension_indices)),
      dimension_indices_(dimension_indices) {}

DiffTensorIndex::DiffTensorIndex(const Shape& shape, const Array& dim_diffs)
    : shape_(shape), dimension_diffs_(dim_diffs) {
  CHECK(dimension_diffs_.size() == shape.DimensionCount())
      << "Dimension indices do not have the same number of dimensions as the "
         "supplied shape!";
  for (int i = 0; i < dimension_diffs_.size(); ++i) {
    CHECK(std::abs(dimension_diffs_[i]) <= shape[i])
        << "Out of bounds dimension index";
  }
}

TensorIndex DiffTensorIndex::CyclicAdd(const TensorIndex& ti) const {
  CHECK(shape_ == ti.GetShape());
  std::vector<int> dim_indices;
  for (int dim = 0; dim < shape_.DimensionCount(); ++dim) {
    int dim_idx = ti[dim] + dimension_diffs_[dim];
    while (dim_idx < 0) {
      dim_idx += shape_[dim];
    }
    dim_idx %= shape_[dim];
    dim_indices.push_back(dim_idx);
  }
  return TensorIndex(ti.GetShape(), dim_indices);
}

std::optional<TensorIndex> DiffTensorIndex::NonCyclicAdd(
    const TensorIndex& ti) const {
  CHECK(shape_ == ti.GetShape());
  std::vector<int> dim_indices;
  for (int dim = 0; dim < shape_.DimensionCount(); ++dim) {
    int dim_idx = ti[dim] + dimension_diffs_[dim];
    if (dim_idx >= shape_[dim] || dim_idx < 0) {
      return std::nullopt;
    }
    dim_indices.push_back(dim_idx);
  }
  return TensorIndex(shape_, dim_indices);
}

}  // namespace fhelipe
