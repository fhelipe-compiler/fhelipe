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

#ifndef FHELIPE_DICTIONARY_H_
#define FHELIPE_DICTIONARY_H_

#include <atomic>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ct_op.h"
#include "io_utils.h"

namespace fhelipe {

using KeyType = std::string;

template <class ValueType>
class Dictionary {
 public:
  typedef std::unordered_map<std::string,
                             std::unique_ptr<Dictionary> (*)(std::istream&)>
      DerivedRecordType;
  virtual void Record(const KeyType& key, const ValueType& value) = 0;
  KeyType Record(const ValueType& value);
  virtual bool Contains(const KeyType& key) const = 0;
  // nsamar: Must be return by value here (and NOT return by reference) because
  // the PersistedDictionary does not maintain the value after the return
  // (that's its entire purpose)
  virtual ValueType At(const KeyType& key) const = 0;
  virtual const std::set<KeyType>& Keys() const = 0;
  virtual ~Dictionary() = default;
  virtual std::unique_ptr<Dictionary<ValueType>> CloneUniq() const = 0;
  virtual void WriteStreamHelper(std::ostream& stream) const = 0;
  static std::unique_ptr<Dictionary> CreateInstance(std::istream& stream);

 protected:
  static DerivedRecordType& GetMap() {
    static DerivedRecordType record_map_;
    return record_map_;
  }
};

template <class T>
struct ReadStreamWithoutTypeNamePrefixImpl {
  static T ReadStreamWithoutTypeNamePrefixFunc(std::istream& stream);
};

template <class DerivedT, class T>
std::enable_if_t<std::is_base_of_v<Dictionary<T>, DerivedT>, DerivedT>
ReadStreamWithoutTypeNamePrefix(std::istream& stream) {
  return ReadStreamWithoutTypeNamePrefixImpl<
      DerivedT>::ReadStreamWithoutTypeNamePrefixFunc(stream);
}

template <class DerivedT, class T>
inline std::enable_if_t<std::is_base_of_v<Dictionary<T>, DerivedT>, T>
ReadStream(std::istream& stream) {
  auto token = ReadStream<std::string>(stream);
  CHECK(token == T::TypeName());
  return ReadStreamWithoutTypeNamePrefix<T>(stream);
}

template <class ValueType>
KeyType Dictionary<ValueType>::Record(const ValueType& value) {
  static std::atomic<int> curr_idx = 0;
  auto key = "__" + std::to_string(++curr_idx);
  Record(key, value);
  return key;
}

template <class ValueType>
std::vector<ValueType> GetValues(const Dictionary<ValueType>& dict) {
  std::vector<ValueType> result;
  for (const auto& key : dict.Keys()) {
    result.push_back(dict.At(key));
  }
  return result;
}

template <class ValueType>
void AppendDictionary(Dictionary<ValueType>& lhs,
                      const Dictionary<ValueType>& append_me) {
  for (const auto& key : append_me.Keys()) {
    lhs.Record(key, append_me.At(key));
  }
}

template <class T>
struct IoStreamImpl<Dictionary<T>> {
  static void WriteStreamFunc(std::ostream& stream, const Dictionary<T>& dict) {
    dict.WriteStreamHelper(stream);
  }
};

}  // namespace fhelipe

#endif  // FHELIPE_DICTIONARY_H_
