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

#ifndef FHELIPE_PROGRAM_CONTEXT_GFLAG_UTILS_H_
#define FHELIPE_PROGRAM_CONTEXT_GFLAG_UTILS_H_

#include <filesystem>

#include "include/constants.h"
#include "include/program_context.h"

DEFINE_int32(log_chunk_size, fhelipe::kDefaultLogChunkSize.value(),
             "Log of ciphertext chunk size (i.e., number of slots)");
DEFINE_int32(log_scale, fhelipe::kDefaultLogScale, "LogScale of ciphertexts");
DEFINE_int32(usable_levels, fhelipe::kDefaultUsableLevels,
             "Max usable levels after bootstarpping");
DEFINE_int32(bootstrapping_precision,
             fhelipe::kDefaultBootstrappingPrecision.value(),
             "Bit-precision of bootstrapping");

namespace {

fhelipe::ProgramContext ProgramContextFromFlags() {
  return {FLAGS_log_chunk_size, FLAGS_log_scale, FLAGS_usable_levels,
          FLAGS_bootstrapping_precision};
}

}  // namespace

#endif  // FHELIPE_PROGRAM_CONTEXT_GFLAG_UTILS_H_
