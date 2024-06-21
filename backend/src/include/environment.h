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

#ifndef FHELIPE_ENVIRONMENT_H_
#define FHELIPE_ENVIRONMENT_H_

#include <pwd.h>
#include <unistd.h>

#include <string>

namespace fhelipe::env {

static std::string Username() {
  // nsamar: Nasty hack that makes it impossible for anyone else to run code
  // safely
  return "nsamar";
}

}  // namespace fhelipe::env

#endif  // FHELIPE_ENVIRONMENT_H_
