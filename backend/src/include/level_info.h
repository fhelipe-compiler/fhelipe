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

#ifndef FHELIPE_LEVEL_INFO_H_
#define FHELIPE_LEVEL_INFO_H_

#include <ostream>

#include "include/io_utils.h"
#include "include/level.h"
#include "include/log_scale.h"

namespace fhelipe {

class LevelInfo {
 public:
  LevelInfo& operator=(const LevelInfo& other) = default;
  LevelInfo(Level level, LogScale log_scale)
      : level_(level), log_scale_(log_scale) {}
  Level Level() const { return level_; }
  LogScale LogScale() const { return log_scale_; }

 private:
  class Level level_;
  class LogScale log_scale_;
};

template <>
void WriteStream<LevelInfo>(std::ostream& stream, const LevelInfo& level_info);

template <>
LevelInfo ReadStream<LevelInfo>(std::istream& stream);

inline std::ostream& operator<<(std::ostream& stream,
                                const LevelInfo& level_info) {
  WriteStream<LevelInfo>(stream, level_info);
  return stream;
}

inline bool operator==(const LevelInfo& lhs, const LevelInfo& rhs) {
  return lhs.Level() == rhs.Level() && lhs.LogScale() == rhs.LogScale();
}

}  // namespace fhelipe

#endif  // FHELIPE_LEVEL_INFO_H_
