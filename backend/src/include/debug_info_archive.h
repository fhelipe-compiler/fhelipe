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

#ifndef FHELIPE_DEBUG_INFO_ARCHIVE_H_
#define FHELIPE_DEBUG_INFO_ARCHIVE_H_

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>

#include "extended_std.h"
#include "io_utils.h"

namespace fhelipe {

class SourceSet {
 public:
  SourceSet() = default;
  explicit SourceSet(const std::set<int>& values);
  bool Contains(int value) const;
  const std::set<int>& Values() const;

 private:
  std::set<int> values_;
};

template <>
void WriteStream<SourceSet>(std::ostream& stream, const SourceSet& src);

template <>
SourceSet ReadStream<SourceSet>(std::istream& stream);

bool operator==(const SourceSet& lhs, const SourceSet& rhs);

}  // namespace fhelipe

namespace std {

template <>
struct hash<fhelipe::SourceSet> {
  std::size_t operator()(const fhelipe::SourceSet& src_set) const {
    return *(src_set.Values().begin());
  }
};

};  // namespace std

namespace fhelipe {

class DestinationSet {
 public:
  DestinationSet() = default;
  explicit DestinationSet(const std::set<int>& values) : values_(values) {}
  bool Contains(int value) const { return Estd::contains(values_, value); }
  const std::set<int>& Values() const { return values_; }

 private:
  std::set<int> values_;
};

class SetBijection {
 public:
  SetBijection() = default;

  void AddMapping(const SourceSet& src, const DestinationSet& dest);

  std::optional<SourceSet> SetContaining(int value) const {
    for (const auto& [src_set, dest_set] : bijection_) {
      if (src_set.Contains(value)) {
        return src_set;
      }
    }
    return std::nullopt;
  }

  void MergeBijectionsWith(std::vector<SourceSet> sets_to_merge,
                           const SourceSet& src, const DestinationSet& dest);

  std::vector<SourceSet> SetsIntersecting(const SourceSet& values) const {
    std::vector<SourceSet> result;
    for (int value : values.Values()) {
      auto candidate = SetContaining(value);
      if (candidate.has_value()) {
        result.push_back(candidate.value());
      }
    }
    return result;
  }

  const DestinationSet& At(const SourceSet& src_set) const {
    return bijection_.at(src_set);
  }
  const std::unordered_map<SourceSet, DestinationSet>& Mappings() const {
    return bijection_;
  }

 private:
  std::unordered_map<SourceSet, DestinationSet> bijection_;
};

template <>
void WriteStream<DestinationSet>(std::ostream& stream,
                                 const DestinationSet& src);

template <>
DestinationSet ReadStream<DestinationSet>(std::istream& stream);

class DebugInfoArchive {
 public:
  DebugInfoArchive() = default;
  explicit DebugInfoArchive(
      const std::unordered_map<int, std::vector<int>>& backward_ptrs)
      : backward_ptrs_(backward_ptrs) {}

  const std::vector<int>& Mapping(int key) const {
    return backward_ptrs_.at(key);
  }

  void AddMapping(int key, const std::vector<int>& value) {
    CHECK(!Estd::contains_key(backward_ptrs_, key));
    backward_ptrs_.emplace(key, value);
  }

  const std::unordered_map<int, std::vector<int>>& Mappings() const {
    return backward_ptrs_;
  }

 private:
  std::unordered_map<int, std::vector<int>> backward_ptrs_;
};

template <>
void WriteStream<DebugInfoArchive>(std::ostream& stream,
                                   const DebugInfoArchive& src);

template <>
DebugInfoArchive ReadStream<DebugInfoArchive>(std::istream& stream);

DebugInfoArchive MergeAdjacent(const DebugInfoArchive& lhs,
                               const DebugInfoArchive& rhs);

SetBijection ClusterDebugInfoArchive(const DebugInfoArchive& debug_info);

}  // namespace fhelipe

#endif  // FHELIPE_DEBUG_INFO_ARCHIVE_H_
