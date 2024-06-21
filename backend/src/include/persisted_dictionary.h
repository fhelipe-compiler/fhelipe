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

#ifndef FHELIPE_PERSISTED_DICTIONARY_H_
#define FHELIPE_PERSISTED_DICTIONARY_H_

#include <filesystem>
#include <iostream>
#include <set>

#include "ct_op.h"
#include "dictionary.h"
#include "filesystem_utils.h"
#include "glog_flag_avoid_writes.h"

namespace fhelipe {

template <class ValueType>
class PersistedDictionary : public Dictionary<ValueType> {
 public:
  explicit PersistedDictionary(const std::filesystem::path& folder_path);

  void Record(const KeyType& key, const ValueType& value) final;
  bool Contains(const KeyType& key) const final;
  ValueType At(const KeyType& key) const final;
  const std::set<KeyType>& Keys() const final { return keys_; }
  std::unique_ptr<Dictionary<ValueType>> CloneUniq() const final {
    return std::make_unique<PersistedDictionary<ValueType>>(*this);
  }
  void WriteStreamHelper(std::ostream& stream) const final;
  const std::filesystem::path& FolderPath() const { return folder_path_; }
  static const std::string& StaticTypeName() {
    static std::string type_name_ = "PersistedDictionary";
    return type_name_;
  }

 private:
  std::filesystem::path folder_path_;
  std::set<KeyType> keys_;
};

template <class ValueType>
PersistedDictionary<ValueType>::PersistedDictionary(
    const std::filesystem::path& folder_path)
    : folder_path_(folder_path) {
  EnsureDirectoryExists(folder_path_);
  auto contained_files = ContainedFilepaths(folder_path);
  for (const auto& filepath : contained_files) {
    keys_.insert(filepath.filename());
  }
}

template <class ValueType>
struct ReadStreamWithoutTypeNamePrefixImpl<PersistedDictionary<ValueType>> {
  static PersistedDictionary<ValueType> ReadStreamWithoutTypeNamePrefixFunc(
      std::istream& stream) {
    auto dict_path = ReadStream<std::filesystem::path>(stream);
    return PersistedDictionary<ValueType>(dict_path);
  }
};

template <class ValueType>
void PersistedDictionary<ValueType>::WriteStreamHelper(
    std::ostream& stream) const {
  WriteStream<std::string>(stream,
                           PersistedDictionary<ValueType>::StaticTypeName());
  stream << ' ';
  WriteStream<std::string>(stream, FolderPath());
}

template <class ValueType>
PersistedDictionary<ValueType> ClearedPersistedDictionary(
    const std::filesystem::path& folder_path) {
  EnsureDoesNotExist(folder_path);
  return PersistedDictionary<ValueType>(folder_path);
}

template <class ValueType>
bool PersistedDictionary<ValueType>::Contains(const KeyType& key) const {
  return keys_.contains(key);
}

template <class ValueType>
void PersistedDictionary<ValueType>::Record(const KeyType& key,
                                            const ValueType& value) {
  CHECK(!Contains(key)) << key;
  keys_.insert(key);
  if (!AvoidWrites()) {
    WriteFile<ValueType>(folder_path_ / key, value);
  }
}

template <class ValueType>
ValueType PersistedDictionary<ValueType>::At(const KeyType& key) const {
  CHECK(Contains(key)) << folder_path_ << " " << key;
  return ReadFile<ValueType>(folder_path_ / key);
}

}  // namespace fhelipe

#endif  // FHELIPE_PERSISTED_DICTIONARY_H_
