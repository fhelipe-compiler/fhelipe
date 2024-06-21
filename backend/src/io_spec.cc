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

#include "include/io_spec.h"

#include <string>

#include "include/io_utils.h"

namespace fhelipe {

IoSpec FilenameToIoSpec(const std::string& filename) {
  std::stringstream ss(filename);
  std::string name;
  std::getline(ss, name);
  std::size_t delim = name.rfind('_');
  CHECK(delim != std::string::npos);
  auto offset = std::stoi(name.substr(delim + 1));
  name = name.substr(0, delim);
  return {name, offset};
}

}  // namespace fhelipe
