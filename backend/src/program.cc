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

#include "include/program.h"

#include "include/constants.h"
#include "include/filesystem_utils.h"

namespace fhelipe {

std::string Program::Code() const {
  return ReadFileContent(exe_folder_ / kDslOutput);
}

template <>
void WriteStream<Program>(std::ostream& stream, const Program& program) {
  WriteStream(stream, program.ExeFolder());
}

template <>
Program ReadStream<Program>(std::istream& stream) {
  return Program{ReadStream<std::filesystem::path>(stream)};
}

}  // namespace fhelipe
