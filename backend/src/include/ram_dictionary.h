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

#ifndef FHELIPE_RAM_DICTIONARY_H_
#define FHELIPE_RAM_DICTIONARY_H_

#include <glog/logging.h>

#include <functional>
#include <unordered_map>

#include "dictionary.h"
#include "extended_std.h"

namespace fhelipe {

template <class ValueType>
class RamDictionary : public Dictionary<ValueType> {
 public:
  RamDictionary() = default;
  explicit RamDictionary(const std::unordered_map<KeyType, ValueType>& dict)
      : map_(dict), keys_(Estd::extract_keys(map_)) {}

  void Record(const KeyType& key, const ValueType& value) final;
  bool Contains(const KeyType& key) const final;
  ValueType At(const KeyType& key) const final;
  const std::set<KeyType>& Keys() const final { return keys_; }
  void WriteStreamHelper(std::ostream& stream) const final;
  std::unique_ptr<Dictionary<ValueType>> CloneUniq() const final {
    return std::make_unique<RamDictionary<ValueType>>(*this);
  }
  static const std::string& StaticTypeName() {
    static std::string type_name_ = "RamDictionary";
    return type_name_;
  }

 private:
  std::unordered_map<KeyType, ValueType> map_;
  std::set<KeyType> keys_;
};

template <class ValueType>
bool RamDictionary<ValueType>::Contains(const KeyType& key) const {
  return map_.find(key) != map_.end();
}

template <class ValueType>
void RamDictionary<ValueType>::Record(const KeyType& key,
                                      const ValueType& value) {
  CHECK(!Contains(key));
  map_.emplace(key, value);
  keys_.insert(key);
}

template <class ValueType>
void RamDictionary<ValueType>::WriteStreamHelper(std::ostream& stream) const {
  WriteStream<std::string>(stream, RamDictionary<ValueType>::StaticTypeName());
  stream << ' ';
  WriteStream<std::unordered_map<KeyType, ValueType>>(stream, map_);
}

template <class ValueType>
ValueType RamDictionary<ValueType>::At(const KeyType& key) const {
  CHECK(Contains(key));
  return map_.at(key);
}

template <class OutType, class InType>
RamDictionary<OutType> Convert(const Dictionary<InType>& dict,
                               const std::function<OutType(InType)>& func) {
  RamDictionary<OutType> result;
  for (const auto& key : dict.Keys()) {
    result.Record(key, func(dict.At(key)));
  }
  return result;
}

template <class ValueType>
struct ReadStreamWithoutTypeNamePrefixImpl<RamDictionary<ValueType>> {
  static RamDictionary<ValueType> ReadStreamWithoutTypeNamePrefixFunc(
      std::istream& stream) {
    return RamDictionary<ValueType>(
        ReadStream<std::unordered_map<KeyType, ValueType>>(stream));
  }
};

}  // namespace fhelipe

#endif  // FHELIPE_RAM_DICTIONARY_H_
