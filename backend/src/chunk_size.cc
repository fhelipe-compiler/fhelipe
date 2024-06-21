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

#include "include/chunk_size.h"

#include <glog/logging.h>

#include "framework/nominal.h"
#include "include/chunk_ir.h"
#include "include/plaintext.h"
#include "include/utils.h"

namespace {

constexpr int MinLogChunkSize = 0;
constexpr int MaxLogChunkSize = 18;

}  // namespace

namespace fhelipe {

LogChunkSize::LogChunkSize(int log_chunk_size)
    : Nominal<PtVal, int>(log_chunk_size) {
  CHECK(value() >= MinLogChunkSize && value() < MaxLogChunkSize);
}

LogN::LogN(int log_n) : Nominal<LogNTypeIsolation, int>(log_n) {}

ChunkSize::ChunkSize(int chunk_size) : Nominal<PtVal, int>(chunk_size) {
  CHECK(value() >= (1 << MinLogChunkSize) && value() < (1 << MaxLogChunkSize))
      << value();
  CHECK(IsPowerOfTwo(value()));
}

}  // namespace fhelipe
