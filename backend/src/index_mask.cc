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

#include "include/index_mask.h"

namespace fhelipe {

std::vector<int> MaskedIndices(IndexMask mask) {
  std::vector<int> result;
  result.reserve(mask.count());

  for (int i = 0; i < kIndexBits; ++i) {
    if (mask[i]) {
      result.push_back(i);
    }
  }
  return result;
}

IndexMask MaxIndexMask(int size) {
  // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  unsigned long max_index = size - 1;
  max_index |= max_index >> 1;
  max_index |= max_index >> 2;
  max_index |= max_index >> 4;
  max_index |= max_index >> 8;
  max_index |= max_index >> 16;
  return IndexMask(max_index);
}

}  // namespace fhelipe
