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

#include "include/input_c.h"

#include <ostream>

#include "include/ct_op_visitor.h"
#include "include/io_spec.h"

namespace fhelipe {

void InputC::WriteToStream(std::ostream& stream) const {
  stream << "InputC ";
  WriteStream(stream, io_spec_);
  WriteStream<std::string>(stream, " ");
}

}  // namespace fhelipe
