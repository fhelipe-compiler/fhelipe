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

#ifndef FHELIPE_PROGRAM_H_
#define FHELIPE_PROGRAM_H_

#include <filesystem>

#include "io_utils.h"

namespace fhelipe {

class Program {
 public:
  explicit Program(const std::filesystem::path& exe_folder)
      : exe_folder_(exe_folder) {}

  std::string Code() const;
  const std::filesystem::path& ExeFolder() const { return exe_folder_; }

 private:
  std::filesystem::path exe_folder_;
};

template <>
void WriteStream<Program>(std::ostream& stream, const Program& program);

template <>
Program ReadStream<Program>(std::istream& stream);

}  // namespace fhelipe

#endif  // FHELIPE_PROGRAM_H_
