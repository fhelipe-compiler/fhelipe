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

#include "include/level_info.h"

#include <ostream>

#include "include/io_utils.h"
#include "include/level.h"

namespace fhelipe {

template <>
void WriteStream<LevelInfo>(std::ostream& stream, const LevelInfo& level_info) {
  WriteStream<Level>(stream, level_info.Level());
  stream << " " << level_info.LogScale().value() << " ";
}

template <>
LevelInfo ReadStream<LevelInfo>(std::istream& stream) {
  auto level = ReadStream<int>(stream);
  auto log_scale = ReadStream<int>(stream);
  return {level, log_scale};
}

}  // namespace fhelipe
