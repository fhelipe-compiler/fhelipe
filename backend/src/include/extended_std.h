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

#ifndef FHELIPE_EXTENDED_STD_H_
#define FHELIPE_EXTENDED_STD_H_

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Estd {

template <typename T>
std::set<T> set_union(const std::set<T>& lhs, const std::set<T>& rhs) {
  std::set<T> result;
  std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                 std::inserter(result, result.begin()));
  return result;
}

template <typename T>
std::set<T> set_intersection(const std::set<T>& lhs, const std::set<T>& rhs) {
  std::set<T> result;
  std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                        std::inserter(result, result.begin()));
  return result;
}

template <typename T>
std::vector<T> concat(std::vector<T> first, const std::vector<T> second) {
  first.insert(first.end(), second.begin(), second.end());
  return first;
}

template <typename T>
std::vector<T> reverse(std::vector<T> vec) {
  std::reverse(vec.begin(), vec.end());
  return vec;
}

inline std::vector<int> index_range(int range_begin, int range_end) {
  std::vector<int> result;
  for (int i = range_begin; i < range_end; ++i) {
    result.push_back(i);
  }
  return result;
}

template <class T>
std::set<T> vector_to_set(const std::vector<T>& in_set) {
  std::set<T> result;
  for (const T& value : in_set) {
    result.insert(value);
  }
  return result;
}

template <typename T>
void sort(std::vector<T>& vec) {
  std::sort(vec.begin(), vec.end());
}

template <class T>
std::vector<T> set_to_vector(const std::set<T>& in_set) {
  std::vector<T> result;
  for (const T& value : in_set) {
    result.push_back(value);
  }
  return result;
}

template <class KeyType, class ValueType>
bool contains_key(const std::unordered_map<KeyType, ValueType>& map,
                  const KeyType& key) {
  return map.find(key) != map.end();
}

template <class KeyType, class ValueType>
bool contains_key(const std::unordered_map<const KeyType*, ValueType>& map,
                  const KeyType* key) {
  return map.find(key) != map.end();
}

template <class KeyType, class ValueType>
const KeyType& find_key(const std::unordered_map<KeyType, ValueType>& map,
                        const ValueType& value) {
  for (auto it = map.begin(); it != map.end(); ++it) {
    if (it->second == value) {
      return it->first;
    }
  }
  LOG(FATAL);
}

template <class KeyType, class ValueType>
std::set<KeyType> extract_keys(
    const std::unordered_map<KeyType, ValueType>& map) {
  std::set<KeyType> result;
  for (const auto& key_value_pair : map) {
    result.insert(key_value_pair.first);
  }
  return result;
}

template <typename Container, typename FuncType>
bool any_of(const Container& container, const FuncType& predicate) {
  return std::any_of(container.begin(), container.end(), predicate);
}

template <typename Container, typename FuncType>
bool all_of(const Container& container, const FuncType& predicate) {
  return std::all_of(container.begin(), container.end(), predicate);
}

template <typename Container, typename FuncType>
bool none_of(const Container& container, const FuncType& predicate) {
  return std::none_of(container.begin(), container.end(), predicate);
}

template <class T>
std::set<T> set_difference(const std::set<T>& lhs, const std::set<T>& rhs) {
  std::set<T> result;
  std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                      std::inserter(result, result.end()));
  return result;
}

template <class T>
bool all_equal(const std::vector<T>& values) {
  return std::adjacent_find(values.begin(), values.end(),
                            std::not_equal_to<>()) == values.end();
}

template <typename KeyType, typename ValueType>
std::vector<ValueType> values_from_keys(
    const std::unordered_map<KeyType, ValueType>& map,
    const std::vector<KeyType>& key_vec) {
  std::vector<ValueType> values;
  values.reserve(key_vec.size());
  for (const auto& key : key_vec) {
    values.push_back(map.at(key));
  }
  return values;
}

template <typename KeyType, typename ValueType>
std::vector<ValueType> values_from_keys(
    const std::unordered_map<const KeyType*, ValueType>& map,
    const std::vector<std::shared_ptr<KeyType>>& key_vec) {
  std::vector<ValueType> values;
  values.reserve(key_vec.size());
  for (const auto& key : key_vec) {
    values.push_back(map.at(key.get()));
  }
  return values;
}

template <typename KeyType, typename ValueType>
std::vector<ValueType> get_values(
    const std::unordered_map<KeyType, ValueType>& map) {
  std::vector<ValueType> result;
  for (const auto& [key, value] : map) {
    result.push_back(value);
  }
  return result;
}

template <class ContainerType, template <typename...> class IndexContainerType>
std::vector<int> permute(const ContainerType& vec,
                         const IndexContainerType<int>& order) {
  std::vector<int> result;
  for (int curr_idx : order) {
    result.push_back(vec[curr_idx]);
  }
  return result;
}

template <template <typename...> class ContainerType, typename T>
ContainerType<T> shuffle(const ContainerType<T>& container) {
  ContainerType<T> result = container;
  static auto rng = std::default_random_engine{};
  std::shuffle(result.begin(), result.end(), rng);
  return result;
}

inline std::vector<int> indices(int upto) {
  std::vector<int> result(upto);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

inline std::vector<int> indices(int from, int upto) {
  std::vector<int> result(upto - from);
  std::iota(result.begin(), result.end(), from);
  return result;
}

template <template <typename...> class ContainerType, typename T,
          typename FuncType>
ContainerType<T> filter(const ContainerType<T>& container, FuncType predicate) {
  ContainerType<T> result;
  std::copy_if(container.begin(), container.end(),
               std::inserter(result, result.end()), predicate);
  return result;
}

bool is_substring(const std::string& supstr, const std::string& substr);

template <typename InType0, typename InType1, typename OutType, int size>
std::array<OutType, size> transform(std::array<InType0, size> in_array_0,
                                    const std::array<InType1, size>& in_array_1,
                                    std::function<int(int, int)> func) {
  std::transform(in_array_0.begin(), in_array_0.end(), in_array_1.begin(),
                 in_array_0.begin(), func);
  return in_array_0;
}

template <typename InType, typename OutType, int size, typename FuncType>
std::array<OutType, size> transform(const std::array<InType, size>& in_array,
                                    FuncType func) {
  std::array<OutType, size> result;
  std::transform(in_array.begin(), in_array.end(), std::back_inserter(result),
                 func);
  return result;
}

template <typename InType0, typename InType1, typename FuncType>
void for_each(const std::vector<InType0>& in_vector_0,
              const std::vector<InType1>& in_vector_1, FuncType func) {
  CHECK(in_vector_0.size() == in_vector_1.size());
  for (const auto& idx : indices(in_vector_0.size())) {
    func(in_vector_0[idx], in_vector_1[idx]);
  }
}

template <typename InType0, typename InType1, typename FuncType>
std::vector<InType0> transform(std::vector<InType0> in_vector_0,
                               const std::vector<InType1>& in_vector_1,
                               FuncType func) {
  CHECK(in_vector_0.size() == in_vector_1.size());
  std::transform(in_vector_0.begin(), in_vector_0.end(), in_vector_1.begin(),
                 in_vector_0.begin(), func);
  return in_vector_0;
}

template <template <typename...> class ContainerType, typename T,
          typename FuncType>
void for_each(const ContainerType<T>& in_vector, FuncType func) {
  std::for_each(in_vector.begin(), in_vector.end(), func);
}

template <template <typename...> class ContainerType, typename T,
          typename FuncType>
std::vector<std::invoke_result_t<FuncType, T>> transform(
    const ContainerType<T>& in_container, FuncType func) {
  std::vector<std::invoke_result_t<FuncType, T>> result;
  result.reserve(in_container.size());
  std::transform(in_container.begin(), in_container.end(),
                 std::back_inserter(result), func);
  return result;
}

template <typename T, typename FuncType>
std::set<std::invoke_result_t<FuncType, T>> transform(
    const std::set<T>& container, FuncType func) {
  std::set<std::invoke_result_t<FuncType, T>> result;
  for (const auto& value : container) {
    result.insert(func(value));
  }
  return result;
}

template <typename T>
void append(std::vector<T>& append_to_me,
            const std::vector<T>& append_from_me) {
  append_to_me.insert(append_to_me.end(), append_from_me.begin(),
                      append_from_me.end());
}

template <typename T>
int find_index(const std::vector<T>& vec, const T& value) {
  return std::find(vec.begin(), vec.end(), value) - vec.begin();
}

template <typename T, typename Func>
int find_index_pred(const std::vector<T>& vec, const Func& pred) {
  return std::find_if(vec.begin(), vec.end(), pred) - vec.begin();
}

template <template <typename...> class Container, typename T>
T sum(const Container<T>& container) {
  return std::accumulate(container.begin(), container.end(), T(0));
}

template <typename Container0, typename Container1>
bool is_equal_as_sets(const Container0& c0, const Container1& c1) {
  return std::set(c0.begin(), c0.end()) == std::set(c1.begin(), c1.end());
}

template <typename Container, typename T>
int count(const Container& container, const T& value) {
  return std::count(container.begin(), container.end(), value);
}

template <typename Container, typename FuncT>
int count_if(const Container& container, const FuncT& predicate) {
  return std::count_if(container.begin(), container.end(), predicate);
}

template <typename Container, typename T>
bool contains(const Container& container, const T& value) {
  return std::find(container.begin(), container.end(), value) !=
         container.end();
}

template <typename T>
int argmin(const std::vector<T>& container) {
  return std::min_element(container.begin(), container.end()) -
         container.begin();
}

template <typename T>
int argmax(const std::vector<T>& container) {
  return std::max_element(container.begin(), container.end()) -
         container.begin();
}

template <typename T>
const T& min_element(const std::vector<T>& container) {
  CHECK(!container.empty());
  return *std::min_element(container.begin(), container.end());
}

template <typename T>
const T& min_element(const std::set<T>& container) {
  CHECK(!container.empty());
  return *container.begin();
}

template <typename T>
const T& max_element(const std::vector<T>& container) {
  CHECK(!container.empty());
  return *std::max_element(container.begin(), container.end());
}

template <typename T>
const T& max_element(const std::set<T>& container) {
  CHECK(!container.empty());
  return *container.rbegin();
}

template <typename T>
T closest_element_less_than_or_equal_to(const std::vector<T>& values,
                                        const T& value) {
  CHECK(!values.empty());
  T closest = values.at(Estd::find_index_pred(
      values, [&value](const T& x) { return x <= value; }));
  for (const auto& x : values) {
    if (x > closest && x <= value) {
      closest = x;
    }
  }
  CHECK(closest <= value);
  return closest;
}

}  // namespace Estd

#endif  // FHELIPE_EXTENDED_STD_H_
