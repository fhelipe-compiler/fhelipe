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

#include "include/debug_info_archive.h"

#include <unordered_map>

#include "include/extended_std.h"
#include "include/io_utils.h"

namespace fhelipe {

template <typename T>
T Merge(const T& lhs, const T& rhs) {
  return T{Estd::set_union(lhs.Values(), rhs.Values())};
}

template <typename T>
T Merge(std::vector<T> vec) {
  if (vec.empty()) {
    return T{};
  }
  if (vec.size() == 1) {
    return vec.at(0);
  }
  auto last = vec.back();
  vec.pop_back();
  return Merge<T>(last, Merge(vec));
}

SourceSet::SourceSet(const std::set<int>& values) : values_(values) {}
bool SourceSet::Contains(int value) const {
  return Estd::contains(values_, value);
}
const std::set<int>& SourceSet::Values() const { return values_; }

bool operator==(const SourceSet& lhs, const SourceSet& rhs) {
  return lhs.Values() == rhs.Values();
}

SetBijection ClusterDebugInfoArchive(const DebugInfoArchive& debug_info) {
  SetBijection bijection;
  for (const auto& [dest, srcs] : debug_info.Mappings()) {
    auto src_set = SourceSet(Estd::vector_to_set(srcs));
    auto src_sets_intersecting = bijection.SetsIntersecting(src_set);
    bijection.MergeBijectionsWith(src_sets_intersecting, src_set,
                                  DestinationSet{std::set<int>{dest}});
  }
  return bijection;
}

template <>
DebugInfoArchive IoStreamImpl<DebugInfoArchive>::ReadStreamFunc(
    std::istream& stream) {
  return DebugInfoArchive{
      ReadStream<std::unordered_map<int, std::vector<int>>>(stream)};
}

template <>
void IoStreamImpl<DebugInfoArchive>::WriteStreamFunc(
    std::ostream& stream, const DebugInfoArchive& archive) {
  WriteStream<std::unordered_map<int, std::vector<int>>>(stream,
                                                         archive.Mappings());
}

DebugInfoArchive MergeAdjacent(const DebugInfoArchive& lhs,
                               const DebugInfoArchive& rhs) {
  auto result = DebugInfoArchive();
  for (const auto& [key, mappings] : rhs.Mappings()) {
    std::vector<int> result_mappings;
    for (int second_key : mappings) {
      for (int value : lhs.Mapping(second_key)) {
        result_mappings.push_back(value);
      }
    }
    result.AddMapping(key, result_mappings);
  }
  return DebugInfoArchive{result};
}

void SetBijection::MergeBijectionsWith(std::vector<SourceSet> sets_to_merge,
                                       const SourceSet& src,
                                       const DestinationSet& dest) {
  std::vector<DestinationSet> dest_sets{dest};
  for (const auto& src_set : sets_to_merge) {
    dest_sets.push_back(bijection_.at(src_set));
    bijection_.erase(src_set);
  }
  if (Estd::contains_key(bijection_, src)) {
    bijection_.erase(src);
  }
  sets_to_merge.push_back(src);
  auto src_set = Merge<SourceSet>(sets_to_merge);
  auto dest_set = Merge<DestinationSet>(dest_sets);
  bijection_.emplace(src_set, dest_set);
}

void SetBijection::AddMapping(const SourceSet& src,
                              const DestinationSet& dest) {
  for (const auto& [src_set, dest_set] : bijection_) {
    // Panic if some elements in `src` already in bijection
    for (int elem : src.Values()) {
      CHECK(!Estd::contains(src_set.Values(), elem));
    }
    // Panic if some elements in `dest` already in bijection
    for (int elem : dest.Values()) {
      CHECK(!Estd::contains(dest_set.Values(), elem));
    }
  }

  bijection_.emplace(src, dest);
}

template <>
void WriteStream(std::ostream& stream, const SourceSet& src) {
  WriteStream(stream, src.Values());
}

template <>
SourceSet ReadStream(std::istream& stream) {
  return SourceSet{ReadStream<std::set<int>>(stream)};
}

template <>
void WriteStream(std::ostream& stream, const DestinationSet& src) {
  WriteStream(stream, src.Values());
}

template <>
DestinationSet ReadStream(std::istream& stream) {
  return DestinationSet{ReadStream<std::set<int>>(stream)};
}

template <>
void WriteStream<DebugInfoArchive>(std::ostream& stream,
                                   const DebugInfoArchive& src) {
  WriteStream(stream, src.Mappings());
}

template <>
DebugInfoArchive ReadStream<DebugInfoArchive>(std::istream& stream) {
  return DebugInfoArchive{
      ReadStream<std::unordered_map<int, std::vector<int>>>(stream)};
}

}  // namespace fhelipe
