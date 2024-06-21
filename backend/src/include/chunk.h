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

#ifndef FHELIPE_CHUNK_H_
#define FHELIPE_CHUNK_H_

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "chunk_size.h"
#include "extended_std.h"

namespace fhelipe {

template <class T>
class Chunk {
 public:
  explicit Chunk(const std::vector<T>& values);
  const std::vector<T>& Values() const;
  ChunkSize size() const;

 private:
  std::vector<T> values_;
};

template <class T>
Chunk<T>::Chunk(const std::vector<T>& values) : values_(values) {
  ChunkSize(values_.size());
}

template <class T>
ChunkSize Chunk<T>::size() const {
  return values_.size();
}

template <class T>
const std::vector<T>& Chunk<T>::Values() const {
  return values_;
}

template <class T>
Chunk<T> Add(const Chunk<T>& chunk0, const Chunk<T>& chunk1) {
  CHECK(chunk0.size() == chunk1.size());
  return Chunk<T>(
      Estd::transform(chunk0.Values(), chunk1.Values(), std::plus<>()));
}

template <class T>
Chunk<T> AddScalar(const Chunk<T>& chunk, const T& scalar) {
  return Chunk<T>(Estd::transform(chunk.Values(),
                                  [scalar](const T& x) { return x + scalar; }));
}
template <class T>
Chunk<T> Mul(const Chunk<T>& chunk0, const Chunk<T>& chunk1) {
  return Chunk<T>(
      Estd::transform(chunk0.Values(), chunk1.Values(), std::multiplies<>()));
}
template <class T>
Chunk<T> MulScalar(const Chunk<T>& chunk, const T& scalar) {
  return Chunk<T>(Estd::transform(chunk.Values(),
                                  [scalar](const T& x) { return x * scalar; }));
}

template <class T>
Chunk<T> Rotate(const Chunk<T>& chunk, int rotate_by) {
  while (rotate_by < 0) {
    rotate_by += chunk.size().value();
  }
  rotate_by %= chunk.size().value();
  std::vector<T> values = chunk.Values();
  std::rotate(values.begin(), values.begin() + values.size() - rotate_by,
              values.end());
  return Chunk<T>(values);
}

}  // namespace fhelipe

#endif  // FHELIPE_CHUNK_H_
