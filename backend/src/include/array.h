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

#ifndef FHELIPE_ARRAY_H_
#define FHELIPE_ARRAY_H_

#include <glog/logging.h>

#include <array>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <vector>

#include "io_utils.h"

namespace fhelipe {

namespace {
constexpr int kMaxDimSize = 6;
}

class Array {
 public:
  using Iterator = std::array<int, kMaxDimSize>::iterator;
  using ConstIterator = std::array<int, kMaxDimSize>::const_iterator;

  explicit Array(int sz = 0) : arr_(), size_(sz) {
    for (auto& elem : arr_) {
      elem = 0;
    }
  }

  template <typename InputIt>
  explicit Array(InputIt begin, InputIt end) : arr_(), size_(end - begin) {
    for (auto i = begin; i < end; ++i) {
      arr_[i - begin] = *i;
    }
    for (int i = end - begin; i < kMaxDimSize; ++i) {
      arr_[i] = 0;
    }
  }

  explicit Array(const std::vector<int>& vec) : size_(vec.size()) {
    CHECK(vec.size() <= kMaxDimSize);
    for (int idx = 0; idx < vec.size(); ++idx) {
      arr_[idx] = vec[idx];
    }
  }

  int push_back(int value) {
    CHECK(size_ != kMaxDimSize);
    arr_[size_++] = value;
    return value;
  }

  Iterator begin() { return arr_.begin(); }
  ConstIterator begin() const { return arr_.begin(); }
  Iterator end() { return arr_.begin() + size_; }
  ConstIterator end() const { return arr_.begin() + size_; }

  int operator[](int idx) const {
    CHECK(idx < size_ && idx >= 0);
    return arr_.at(idx);
  }
  int& operator[](int idx) {
    CHECK(idx < size_ && idx >= 0);
    return arr_.at(idx);
  }

  friend bool operator==(const Array& lhs, const Array& rhs);
  int size() const { return size_; }

 private:
  std::array<int, kMaxDimSize> arr_;
  int size_;
};

inline bool operator==(const Array& lhs, const Array& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.arr_.begin(), lhs.arr_.begin() + lhs.size(),
                    rhs.arr_.begin());
}

inline bool operator!=(const Array& lhs, const Array& rhs) {
  return !(lhs == rhs);
}

template <>
inline void WriteStream<Array>(std::ostream& stream, const Array& arr) {
  stream << arr.size() << " ";
  for (int idx = 0; idx < arr.size(); ++idx) {
    stream << arr[idx] << " ";
  }
}

template <>
inline Array ReadStream<Array>(std::istream& stream) {
  auto vec = ReadStream<std::vector<int>>(stream);
  return Array{vec.begin(), vec.end()};
}

}  // namespace fhelipe

namespace Estd {

using fhelipe::Array;

inline Array transform(Array in_array_0, const Array& in_array_1,
                       std::function<int(int, int)> func) {
  std::transform(in_array_0.begin(), in_array_0.end(), in_array_1.begin(),
                 in_array_0.begin(), func);
  return in_array_0;
}

template <typename FuncType>
Array transform(const Array& in_array, FuncType func) {
  Array result;
  std::transform(in_array.begin(), in_array.end(), result.begin(), func);
  return result;
}

}  // namespace Estd

#endif  // FHELIPE_ARRAY_H_
