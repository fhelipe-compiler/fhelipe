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

#ifndef FHELIPE_TENSOR_H_
#define FHELIPE_TENSOR_H_

#include <glog/logging.h>

#include <functional>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

#include "constants.h"
#include "io_utils.h"
#include "plaintext.h"
#include "shape.h"
#include "tensor_index.h"

namespace fhelipe {

template <class T>
class Tensor {
 public:
  Tensor(const Shape& shape, const std::vector<T>& values);

  Shape GetShape() const;

  T& operator[](const TensorIndex& idx);
  T operator[](const TensorIndex& idx) const;
  int DimensionCount() const;
  const std::vector<T> Values() const { return values_; }

 private:
  Shape shape_;
  std::vector<T> values_;
};

template <class T>
inline T& Tensor<T>::operator[](const TensorIndex& idx) {
  CHECK(idx.GetShape() == shape_);
  CHECK(idx.Flat() < values_.size());
  return values_[idx.Flat()];
}

template <class T>
T Tensor<T>::operator[](const TensorIndex& idx) const {
  CHECK(idx.GetShape() == shape_);
  CHECK(idx.Flat() < values_.size());
  return values_[idx.Flat()];
}

template <class T>
Tensor<T>::Tensor(const Shape& shape, const std::vector<T>& values)
    : shape_(shape), values_(values) {
  CHECK(shape_.ValueCnt() == values_.size());
}

template <class T>
Shape Tensor<T>::GetShape() const {
  return shape_;
}

template <class T>
int Tensor<T>::DimensionCount() const {
  return shape_.DimensionCount();
}

template <class T>
struct IoStreamImpl<Tensor<T>> {
  static void WriteStreamFunc(std::ostream& stream, const Tensor<T>& tensor) {
    stream << std::fixed << std::showpoint
           << std::setprecision(kPrintDoublePrecision);
    WriteStream(stream, tensor.GetShape());
    stream << "\n";
    WriteStream(stream, tensor.Values());
  }
  static Tensor<T> ReadStreamFunc(std::istream& stream) {
    auto shape = ReadStream<Shape>(stream);
    auto values = ReadStream<std::vector<T>>(stream);
    return {shape, values};
  }
};

}  // namespace fhelipe

#endif  // FHELIPE_TENSOR_H_
