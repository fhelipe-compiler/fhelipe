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

#include "include/permutation.h"

#include <iostream>

#include "include/io_utils.h"

namespace fhelipe {

Permutation::Permutation(int size) : permutation_(Estd::indices(size)) {}

Permutation::Permutation(const std::vector<int>& permutation)
    : permutation_(permutation) {
  CHECK(Estd::vector_to_set(permutation_) ==
        Estd::vector_to_set(Estd::indices(permutation_.size())));
}

bool operator==(const PermutationCycle& lhs, const PermutationCycle& rhs) {
  return rhs.cycle_ == lhs.cycle_ &&
         lhs.permutation_size_ == rhs.permutation_size_;
}

PermutationCycle::PermutationCycle(int permutation_size,
                                   const std::vector<int>& cycle)
    : permutation_size_(permutation_size), cycle_(cycle) {
  CHECK(Estd::vector_to_set(cycle_).size() == cycle_.size());
  for (int elem : cycle) {
    CHECK(elem >= 0 && elem < permutation_size_);
  }
}

Permutation Permutation::Inverse() const {
  std::vector<int> result(permutation_.size());
  for (int idx : Estd::indices(permutation_.size())) {
    result.at(permutation_.at(idx)) = idx;
  }
  return Permutation{result};
}

bool operator==(const Permutation& lhs, const Permutation& rhs) {
  return lhs.permutation_ == rhs.permutation_;
}

Permutation PermutationCycle::ToPermutation() const {
  auto result = Estd::indices(permutation_size_);
  for (int idx : Estd::indices(cycle_.size())) {
    result.at(cycle_.at((idx + 1) % cycle_.size())) = cycle_.at(idx);
  }
  return Permutation{result};
}

std::pair<PermutationCycle, PermutationCycle> PermutationCycle::BreakUp(
    int breakpoint) const {
  CHECK(breakpoint > 1 && breakpoint < cycle_.size());
  auto first = Estd::transform(Estd::indices(breakpoint),
                               [this](int idx) { return cycle_.at(idx); });
  std::vector<int> second =
      Estd::transform(Estd::indices(breakpoint - 1, cycle_.size()),
                      [this](int idx) { return cycle_.at(idx); });

  return {PermutationCycle{permutation_size_, first},
          PermutationCycle{permutation_size_, second}};
}

namespace {

PermutationCycle FindCycleStartingAt(const Permutation& permutation,
                                     int start_idx) {
  const auto& permutation_vector = permutation.Vector();
  int src_idx = start_idx;
  std::vector<int> cycle;
  do {
    cycle.push_back(src_idx);
    int dest_idx = Estd::find_index(permutation_vector, src_idx);
    src_idx = dest_idx;
  } while (src_idx != start_idx);

  return {permutation.Size(), cycle};
}

}  // namespace

std::vector<PermutationCycle> Permutation::Cycles() const {
  auto identity = Estd::indices(permutation_.size());
  if (permutation_ == identity) {
    return {};
  }

  int first_non_identity = Estd::find_index_pred(
      identity, [this](const int& idx) { return permutation_[idx] != idx; });

  auto cycle = FindCycleStartingAt(*this, first_non_identity);
  auto remaining_permutation = cycle.Apply(permutation_);

  auto result = Permutation(remaining_permutation).Cycles();
  result.push_back(cycle);
  return result;
}

Permutation Compose(const Permutation& lhs, const Permutation& rhs) {
  std::vector<int> result;
  for (int elem : lhs.permutation_) {
    result.push_back(rhs.permutation_.at(elem));
  }
  return Permutation{result};
}

Permutation Compose(std::vector<Permutation> permutations) {
  auto curr = permutations.back();
  permutations.pop_back();
  if (permutations.empty()) {
    return curr;
  }
  return Compose(curr, Compose(permutations));
}

Permutation Compose(const std::vector<PermutationCycle>& cycles) {
  auto perms = Estd::transform(
      cycles, [](const auto& cycle) { return cycle.ToPermutation(); });
  return Compose(perms);
}

void Permutation::WriteStreamHelper(std::ostream& stream) const {
  WriteStream(stream, permutation_);
}

void PermutationCycle::WriteStreamHelper(std::ostream& stream) const {
  WriteStream(stream, permutation_size_);
  stream << " ";
  WriteStream(stream, cycle_);
}

template <>
void WriteStream<Permutation>(std::ostream& stream,
                              const Permutation& permutation) {
  permutation.WriteStreamHelper(stream);
}

template <>
Permutation ReadStream<Permutation>(std::istream& stream) {
  return Permutation{ReadStream<std::vector<int>>(stream)};
}

template <>
void WriteStream<PermutationCycle>(std::ostream& stream,
                                   const PermutationCycle& cycle) {
  cycle.WriteStreamHelper(stream);
}

template <>
PermutationCycle ReadStream<PermutationCycle>(std::istream& stream) {
  auto permutation_size = ReadStream<int>(stream);
  return {permutation_size, ReadStream<std::vector<int>>(stream)};
}

}  // namespace fhelipe
