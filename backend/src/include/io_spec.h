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

#ifndef FHELIPE_IO_SPEC_H_
#define FHELIPE_IO_SPEC_H_

#include <functional>
#include <iosfwd>
#include <string>
#include <system_error>

#include "io_utils.h"

namespace fhelipe {

struct IoSpec {
  std::string name;
  int offset;
  IoSpec(const std::string& in_name, int in_offset)
      : name(in_name), offset(in_offset) {}
};

inline bool operator==(const IoSpec& lhs, const IoSpec& rhs) {
  return lhs.name == rhs.name && lhs.offset == rhs.offset;
}

inline bool operator<(const IoSpec& lhs, const IoSpec& rhs) {
  if (lhs.name == rhs.name) {
    return lhs.offset < rhs.offset;
  }
  return lhs.name < rhs.name;
}

IoSpec FilenameToIoSpec(const std::string& filename);

template <>
inline void WriteStream<IoSpec>(std::ostream& stream, const IoSpec& io_spec) {
  WriteStream(stream, io_spec.name);
  WriteStream<std::string>(stream, " ");
  WriteStream<int>(stream, io_spec.offset);
  WriteStream<std::string>(stream, " ");
}

inline std::ostream& operator<<(std::ostream& stream, const IoSpec& io_spec) {
  WriteStream<IoSpec>(stream, io_spec);
  return stream;
}

inline std::string ToFilename(const IoSpec& io_spec) {
  return io_spec.name + "_" + std::to_string(io_spec.offset);
}

template <>
inline IoSpec ReadStream<IoSpec>(std::istream& iss) {
  auto name = ReadStream<std::string>(iss);
  auto offset = ReadStream<int>(iss);
  return IoSpec(name, offset);
}

}  // namespace fhelipe

template <>
struct std::hash<fhelipe::IoSpec> {
  std::size_t operator()(const fhelipe::IoSpec& k) const {
    return (std::hash<std::string>{}(k.name) ^
            (std::hash<int>{}(k.offset) << 1));
  }
};

#endif  // FHELIPE_IO_SPEC_H_
