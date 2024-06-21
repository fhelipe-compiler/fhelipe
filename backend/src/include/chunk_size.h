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

#ifndef FHELIPE_CHUNK_SIZE_H_
#define FHELIPE_CHUNK_SIZE_H_

#include <istream>

#include "framework/nominal.h"
#include "include/io_utils.h"
#include "include/plaintext.h"
#include "include/utils.h"

namespace fhelipe {

class ChunkSize;

class LogChunkSize : public fwk::Nominal<PtVal, int> {
 public:
  LogChunkSize(int log_chunk_size);
  explicit operator ChunkSize() const;
};

class LogNTypeIsolation;

class LogN : public fwk::Nominal<LogNTypeIsolation, int> {
 public:
  LogN(int log_n);
  explicit LogN(LogChunkSize log_chunk_size)
      : LogN(log_chunk_size.value() + 1) {}
  explicit operator LogChunkSize() const { return value() - 1; }

 private:
};

class ChunkSize : public fwk::Nominal<PtVal, int> {
 public:
  ChunkSize(int chunk_size);
  explicit operator LogChunkSize() const { return Log2(value()); }

 private:
};

inline LogChunkSize::operator ChunkSize() const { return 1 << value(); }

template <>
inline ChunkSize ReadStream<ChunkSize>(std::istream& stream) {
  return {ReadStream<int>(stream)};
}

template <>
inline LogChunkSize ReadStream<LogChunkSize>(std::istream& stream) {
  return {ReadStream<int>(stream)};
}

template <>
inline void WriteStream<ChunkSize>(std::ostream& stream,
                                   const ChunkSize& chunk_size) {
  WriteStream<ChunkSize>(stream, chunk_size.value());
}

template <>
inline void WriteStream<LogChunkSize>(std::ostream& stream,
                                      const LogChunkSize& log_chunk_size) {
  WriteStream<int>(stream, log_chunk_size.value());
}

}  // namespace fhelipe

#endif  // FHELIPE_CHUNK_SIZE_H_
