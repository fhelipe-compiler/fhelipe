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

#include "include/utils.h"

#include <optional>
#include <ostream>
#include <random>
#include <vector>

#include "include/chunk_size.h"
#include "include/dimension_bit.h"
#include "include/io_utils.h"
#include "include/plaintext.h"
#include "include/shape.h"
#include "include/tensor.h"
#include "include/tensor_layout.h"

namespace fhelipe {

double GetUniformRandom() {
  static std::uniform_real_distribution<double> unif(0, 1);
  static std::default_random_engine re;
  return unif(re);
}

}  // namespace fhelipe
