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

#ifndef FHELIPE_SHAPE_H_
#define FHELIPE_SHAPE_H_

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>

#include "array.h"
#include "extended_std.h"
#include "io_utils.h"

namespace fhelipe {

class Shape {
 public:
  using Iterator = std::array<int, kMaxDimSize>::iterator;
  using ConstIterator = std::array<int, kMaxDimSize>::const_iterator;

  Shape() {}

  Shape(const Array& dims) : dimensions_(dims) { CheckDims(); }

  template <typename InputIt>
  explicit Shape(InputIt begin, InputIt end) : dimensions_(begin, end) {
    CheckDims();
  }
  Shape(std::initializer_list<int> l) : dimensions_(l) { CheckDims(); }
  Iterator begin() { return dimensions_.begin(); }
  ConstIterator begin() const { return dimensions_.begin(); }
  Iterator end() { return dimensions_.end(); }
  ConstIterator end() const { return dimensions_.end(); }

  int operator[](int i) const { return dimensions_[i]; }
  int& operator[](int i) { return dimensions_[i]; }

  int DimensionCount() const { return dimensions_.size(); }
  int ValueCnt() const;

  Shape SubShape(int start_i, int end_i) const;
  Shape SubShape(int start_i) const {
    return SubShape(start_i, DimensionCount());
  }
  Shape SpatialShape() const { return SubShape(1); }

  friend bool operator==(const Shape& rhs, const Shape& lhs);

  explicit operator Array() const { return Array(begin(), end()); }

 private:
  Array dimensions_;
  void CheckDims() const;
};

Shape ShapeWithChannels(int channel_cnt, const Shape& spatial_shape);

std::ostream& operator<<(std::ostream& stream, const Shape& shape);

bool IsInRange(const Shape& shape, const Array& indices);

inline bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.dimensions_ == rhs.dimensions_;
}

inline bool operator!=(const Shape& s1, const Shape& s2) { return !(s1 == s2); }

template <>
inline void WriteStream<Shape>(std::ostream& stream, const Shape& shape) {
  std::vector<int> dimensions(shape.begin(), shape.end());
  WriteStream(stream, dimensions);
}

template <>
inline Shape ReadStream<Shape>(std::istream& ir_stream) {
  return {Array(ReadStream<std::vector<int>>(ir_stream))};
}

inline bool operator<(const Shape& lhs, const Shape& rhs) {
  if (lhs.DimensionCount() > rhs.DimensionCount()) {
    return true;
  }
  for (int idx : Estd::indices(lhs.DimensionCount())) {
    if (lhs[idx] < rhs[idx]) {
      return true;
    }
  }
  return false;
}

}  // namespace fhelipe

template <>
struct std::hash<fhelipe::Shape> {
  std::size_t operator()(const fhelipe::Shape& shape) const {
    using std::hash;
    using std::size_t;
    using std::string;

    std::size_t seed = std::hash<int>()(shape.DimensionCount());
    for (auto x : shape) {
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = (x >> 16) ^ x;
      seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

#endif  // FHELIPE_SHAPE_H_
