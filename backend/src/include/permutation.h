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

#ifndef FHELIPE_PERMUTATION_H_
#define FHELIPE_PERMUTATION_H_

#include <vector>

#include "extended_std.h"
#include "io_utils.h"

namespace fhelipe {

class Permutation;

// Permutation cycles use the notation from
// https://en.wikipedia.org/wiki/Cyclic_permutation
// (with the exception that counting starts at 0 and not 1, of course! :) )
// So the permutation
// (1 2 3 0)
// (see Permutation class for permutation notation)
// can be written as a cyclic permutation
// (0 1 2 3)
// or, the index mapping is
// cycle_[(idx+1) % cycle_.size()] -> cycle_[idx]
// and all other indices don't change
class PermutationCycle {
 public:
  PermutationCycle(int permutation_size, const std::vector<int>& cycle);

  std::pair<PermutationCycle, PermutationCycle> BreakUp(int breakpoint) const;
  Permutation ToPermutation() const;

  template <typename T>
  std::vector<T> Apply(const std::vector<T>& values) const;
  int CycleSize() const { return cycle_.size(); }
  int PermutationSize() const { return permutation_size_; }
  friend bool operator==(const PermutationCycle& lhs,
                         const PermutationCycle& rhs);
  void WriteStreamHelper(std::ostream& stream) const;

 private:
  int permutation_size_;
  std::vector<int> cycle_;
};

// Permutations are stored as a vector of int indices, using one-line notation:
// https://en.wikipedia.org/wiki/Permutation
// (with the exception that counting starts at 0 and not 1, of course! :) )
// For example, the permutation
// sigma = (3 4 0 1 2)
// applied on the vector
// 0, 1, 2, 3, 4
// results in the vector
// 3, 4, 0, 1, 2
// That is, the element at position i goes to position sigma[i].
class Permutation {
 public:
  explicit Permutation(const std::vector<int>& permutation);
  explicit Permutation(int size);

  template <typename T>
  std::vector<T> Apply(std::vector<T> values) const;
  const std::vector<int>& Vector() const { return permutation_; }

  void WriteStreamHelper(std::ostream& stream) const;
  int Size() const { return permutation_.size(); }

  std::vector<PermutationCycle> Cycles() const;
  friend Permutation Compose(const Permutation& lhs, const Permutation& rhs);
  Permutation Inverse() const;

  friend bool operator==(const Permutation& lhs, const Permutation& rhs);

 private:
  std::vector<int> permutation_;
};

Permutation Compose(std::vector<Permutation> permutations);
Permutation Compose(const std::vector<PermutationCycle>& permutations);

template <typename T>
std::vector<T> Permutation::Apply(std::vector<T> values) const {
  CHECK(values.size() == permutation_.size());
  std::vector<T> result = values;
  for (int idx : Estd::indices(values.size())) {
    result.at(permutation_.at(idx)) = values.at(idx);
  }
  return result;
}

template <typename T>
std::vector<T> PermutationCycle::Apply(const std::vector<T>& values) const {
  return ToPermutation().Apply(values);
}

template <typename T>
Permutation PermutationFromSourceDestinationPair(const std::vector<T>& src,
                                                 const std::vector<T>& dest) {
  CHECK(src.size() == dest.size());
  CHECK(Estd::vector_to_set(src) == Estd::vector_to_set(dest));
  CHECK(Estd::vector_to_set(src).size() == src.size());
  std::vector<int> permutation;
  for (int idx : Estd::indices(src.size())) {
    permutation.push_back(Estd::find_index(dest, src[idx]));
  }
  return Permutation(permutation);
}

template <>
void WriteStream<Permutation>(std::ostream& stream,
                              const Permutation& permutation);

template <>
Permutation ReadStream<Permutation>(std::istream& stream);

template <>
void WriteStream<PermutationCycle>(std::ostream& stream,
                                   const PermutationCycle& permutation);

template <>
PermutationCycle ReadStream<PermutationCycle>(std::istream& stream);

}  // namespace fhelipe

#endif  // FHELIPE_PERMUTATION_H_
