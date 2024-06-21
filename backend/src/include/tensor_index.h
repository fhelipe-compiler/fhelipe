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

#ifndef TENSOR_INDEX_H_
#define TENSOR_INDEX_H_

#include <iostream>
#include <optional>
#include <vector>

#include "array.h"
#include "io_utils.h"
#include "shape.h"

namespace fhelipe {

class TensorIndex {
 public:
  TensorIndex(const Shape& shape, int flat);
  TensorIndex(const Shape& shape, const Array& dimension_indices);
  TensorIndex(const Shape& shape, const std::vector<int>& dimension_indices);

  int operator[](int idx) const { return dimension_indices_[idx]; }
  Shape GetShape() const;
  int DimensionCount() const { return shape_.DimensionCount(); }
  const Array& DimensionIndices() const { return dimension_indices_; }
  int Flat() const { return flat_index_; }

 private:
  Shape shape_;
  int flat_index_;
  Array dimension_indices_;
};

class DiffTensorIndex {
 public:
  DiffTensorIndex(const Shape& shape, const Array& dim_diffs);
  std::optional<TensorIndex> NonCyclicAdd(const TensorIndex& ti) const;
  TensorIndex CyclicAdd(const TensorIndex& ti) const;

  int operator[](int idx) const { return dimension_diffs_[idx]; }
  int DimensionCount() const { return shape_.DimensionCount(); }
  const Array& DimensionIndices() const { return dimension_diffs_; }
  Shape GetShape() const;
  friend bool operator==(const DiffTensorIndex& lhs,
                         const DiffTensorIndex& rhs);

 private:
  Shape shape_;
  Array dimension_diffs_;
};

inline bool operator==(const DiffTensorIndex& lhs, const DiffTensorIndex& rhs) {
  return lhs.shape_ == rhs.shape_ &&
         lhs.dimension_diffs_ == rhs.dimension_diffs_;
}

inline bool operator<(const TensorIndex& lhs, const TensorIndex& rhs) {
  if (lhs.GetShape() != rhs.GetShape()) {
    return lhs.GetShape() < rhs.GetShape();
  }
  return lhs.Flat() < rhs.Flat();
}

inline Shape TensorIndex::GetShape() const { return shape_; }

inline bool operator==(const TensorIndex& lhs, const TensorIndex& rhs) {
  if (lhs.GetShape() != rhs.GetShape()) {
    return false;
  }
  for (int idx = 0; idx < lhs.DimensionCount(); ++idx) {
    if (lhs[idx] != rhs[idx]) {
      return false;
    }
  }
  return true;
}

inline bool operator!=(const TensorIndex& lhs, const TensorIndex& rhs) {
  return !(lhs == rhs);
}

inline Shape DiffTensorIndex::GetShape() const { return shape_; }

template <>
inline void WriteStream<TensorIndex>(std::ostream& stream,
                                     const TensorIndex& ti) {
  WriteStream<int>(stream, ti.Flat());
}

template <>
inline void WriteStream<DiffTensorIndex>(std::ostream& stream,
                                         const DiffTensorIndex& ti) {
  WriteStream<Shape>(stream, ti.GetShape());
  stream << " ";
  WriteStream<Array>(stream, ti.DimensionIndices());
}

template <>
inline DiffTensorIndex ReadStream<DiffTensorIndex>(std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto indices = ReadStream<Array>(stream);
  return {shape, indices};
}

}  // namespace fhelipe

namespace std {

template <>
struct hash<fhelipe::TensorIndex> {
  std::size_t operator()(const fhelipe::TensorIndex& ti) const {
    using std::hash;
    using std::size_t;
    using std::string;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:

    return (hash<int>()(ti.Flat()));
  }
};

}  // namespace std

#endif  // TENSOR_INDEX_H_
