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

#ifndef FHELIPE_LEVEL_H_
#define FHELIPE_LEVEL_H_

#include "framework/interval.h"
#include "include/io_utils.h"

namespace fhelipe {

class LevelTypeIsolation {};

class Level : public fwk::Interval<LevelTypeIsolation, int, int> {
 public:
  Level(int value) : Interval<LevelTypeIsolation, int, int>(value) {
    CHECK(value >= 1 && value < 100);
  }
  using fwk::Interval<LevelTypeIsolation, int, int>::value;
};

template <>
inline Level ReadStream<Level>(std::istream& stream) {
  return {ReadStream<int>(stream)};
}

template <>
inline void WriteStream<Level>(std::ostream& stream, const Level& level) {
  WriteStream<int>(stream, level.value());
}

}  // namespace fhelipe

#endif  // FHELIPE_LEVEL_H_
