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

#ifndef FHELIPE_UTILS_H_
#define FHELIPE_UTILS_H_

#include <glog/logging.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace fhelipe {

class Shape;

inline int ceil_log2(int value) { return std::ceil(std::log2(value)); }

inline bool IsPowerOfTwo(int n) { return n > 0 && !(n & (n - 1)); }

inline int Log2(int x) {
  CHECK(IsPowerOfTwo(x));
  if (x == 1) {
    return 0;
  }
  return 1 + Log2(x >> 1);
}

}  // namespace fhelipe

#endif  // FHELIPE_UTILS_H_
