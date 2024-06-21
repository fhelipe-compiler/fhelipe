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

#ifndef FHELIPE_PROGRAM_CONTEXT_H_
#define FHELIPE_PROGRAM_CONTEXT_H_

#include "include/bootstrapping_precision.h"
#include "include/chunk_size.h"
#include "include/level.h"
#include "include/log_scale.h"
#include "latticpp/ckks/lattigo_param.h"

namespace fhelipe {

class ProgramContext {
 public:
  ProgramContext(LogChunkSize log_chunk_size, LogScale log_scale,
                 Level usable_levels,
                 BootstrappingPrecision bootstrapping_precision)
      : log_chunk_size_(log_chunk_size),
        log_scale_(log_scale),
        usable_levels_(usable_levels),
        bootstrapping_precision_(bootstrapping_precision) {}

  BootstrappingPrecision GetBootstrappingPrecision() const {
    return bootstrapping_precision_;
  }

  LogChunkSize GetLogChunkSize() const { return log_chunk_size_; }
  LogScale LogScale() const { return log_scale_; }
  Level UsableLevels() const { return usable_levels_; }

  latticpp::LattigoParam GetLattigoParam() const {
    return {GetLogN().value(), log_scale_.value(), usable_levels_.value(),
            bootstrapping_precision_.value()};
  }

  LogN GetLogN() const { return LogN{log_chunk_size_}; }

 private:
  LogChunkSize log_chunk_size_;
  class LogScale log_scale_;
  Level usable_levels_;
  BootstrappingPrecision bootstrapping_precision_;
};

template <>
inline ProgramContext ReadStream<ProgramContext>(std::istream& stream) {
  auto log_chunk_size = ReadStream<LogChunkSize>(stream);
  auto log_scale = ReadStream<int>(stream);
  auto usable_levels = ReadStream<Level>(stream);
  auto bootstrapping_precision = ReadStream<BootstrappingPrecision>(stream);
  return {log_chunk_size, log_scale, usable_levels, bootstrapping_precision};
}

template <>
inline void WriteStream<ProgramContext>(std::ostream& stream,
                                        const ProgramContext& program_context) {
  WriteStream<LogChunkSize>(stream, program_context.GetLogChunkSize());
  stream << " ";
  WriteStream<class LogScale>(stream, program_context.LogScale());
  stream << " ";
  WriteStream<class Level>(stream, program_context.UsableLevels());
  stream << " ";
  WriteStream<BootstrappingPrecision>(
      stream, program_context.GetBootstrappingPrecision());
}

inline ProgramContext MakeProgramContext(const latticpp::LattigoParam& param) {
  return {LogChunkSize{LogN{param.LogN()}}, param.LogScale(),
          param.UsableLevels(), param.BootstrappingPrecision()};
}

}  // namespace fhelipe

#endif  // FHELIPE_PROGRAM_CONTEXT_H_
