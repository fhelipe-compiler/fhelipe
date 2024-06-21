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

#ifndef FHELIPE_IO_UTILS_H_
#define FHELIPE_IO_UTILS_H_

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <istream>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "glog/logging.h"

namespace fhelipe {

static const std::string kInvalidOptionalToken = "#";

// Second parameter needed to specialize IoStreamImpl to all arithmetic types
// genererically via std;:is_arithmetic and enable_if_t
// https://stackoverflow.com/questions/55420989/class-template-specialization-for-multiple-types
template <typename T, typename = void>
struct IoStreamImpl {
  static void WriteStreamFunc(std::ostream& stream, const T& value);
  static T ReadStreamFunc(std::istream& stream);
};

// http://www.gotw.ca/publications/mill17.htm
template <typename T>
void WriteStream(std::ostream& stream, const T& x) {
  IoStreamImpl<T>::WriteStreamFunc(stream, x);
}

template <typename T>
T ReadStream(std::istream& stream) {
  return IoStreamImpl<T>::ReadStreamFunc(stream);
}

template <typename T>
struct IoStreamImpl<
    T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_pointer_v<T>>> {
  static void WriteStreamFunc(std::ostream& stream, const T& x) { stream << x; }
  static T ReadStreamFunc(std::istream& stream) {
    T x;
    stream >> x;
    CHECK(!stream.fail());
    return x;
  }
};

template <>
struct IoStreamImpl<std::string, void> {
  static void WriteStreamFunc(std::ostream& stream, const std::string& x) {
    stream << x;
  }
  static std::string ReadStreamFunc(std::istream& stream) {
    std::string x;
    stream >> x;
    return x;
  }
};

template <typename KeyType, typename ValueType>
struct IoStreamImpl<std::unordered_map<KeyType, ValueType>, void> {
  static void WriteStreamFunc(std::ostream& stream,
                              const std::unordered_map<KeyType, ValueType>& x) {
    WriteStream<int>(stream, x.size());
    stream << '\n';
    for (const auto& [key, value] : x) {
      WriteStream<KeyType>(stream, key);
      stream << ' ';
      WriteStream<ValueType>(stream, value);
      stream << '\n';
    }
  }

  static std::unordered_map<KeyType, ValueType> ReadStreamFunc(
      std::istream& stream) {
    std::unordered_map<KeyType, ValueType> result;
    for (int count = ReadStream<int>(stream); count > 0; count--) {
      auto key = ReadStream<KeyType>(stream);
      auto value = ReadStream<ValueType>(stream);
      result.emplace(key, value);
    }
    return result;
  }
};

template <typename T>
struct IoStreamImpl<std::optional<T>, void> {
  static void WriteStreamFunc(std::ostream& stream, const std::optional<T>& x) {
    if (x.has_value()) {
      WriteStream<T>(stream, x.value());
    } else {
      stream << kInvalidOptionalToken;
    }
  }

  static std::optional<T> ReadStreamFunc(std::istream& stream) {
    auto x = ReadStream<std::string>(stream);
    if (x == kInvalidOptionalToken) {
      return std::nullopt;
    }
    std::stringstream ss(x);
    return ReadStream<T>(ss);
  }
};

template <>
inline std::optional<int> ReadStream<std::optional<int>>(std::istream& stream) {
  auto x = ReadStream<std::string>(stream);
  if (x == kInvalidOptionalToken) {
    return std::nullopt;
  }
  return std::stoi(x);
}

template <class T>
struct IoStreamImpl<std::vector<T>> {
  static void WriteStreamFunc(std::ostream& stream, const std::vector<T>& vec) {
    WriteStream<int>(stream, vec.size());
    stream << " ";
    for (const T& value : vec) {
      WriteStream<T>(stream, value);
      stream << " ";
    }
  }
  static std::vector<T> ReadStreamFunc(std::istream& stream) {
    std::vector<T> result;
    auto vec_size = ReadStream<int>(stream);
    result.reserve(vec_size);
    while (result.size() < vec_size) {
      result.push_back(ReadStream<T>(stream));
    }
    return result;
  }
};

template <class T>
struct IoStreamImpl<std::set<T>> {
  static void WriteStreamFunc(std::ostream& stream, const std::set<T>& values) {
    WriteStream<int>(stream, values.size());
    stream << " ";
    for (const T& value : values) {
      WriteStream<T>(stream, value);
      stream << " ";
    }
  }
  static std::set<T> ReadStreamFunc(std::istream& stream) {
    std::set<T> result;
    auto elem_count = ReadStream<int>(stream);
    while (result.size() < elem_count) {
      result.insert(ReadStream<T>(stream));
    }
    return result;
  }
};

template <typename T>
std::string ToString(const T& value) {
  std::stringstream ss;
  WriteStream<T>(ss, value);
  return ss.str();
}

template <typename T>
std::string DagLabel(const T& value) {
  return ToString(value);
}

template <typename T>
T OpenStream(const std::filesystem::path& filepath) {
  auto stream = T(filepath);
  CHECK(!stream.fail()) << "Unable to open file " << filepath;
  return stream;
}

}  // namespace fhelipe

#endif  // FHELIPE_IO_UTILS_H_
