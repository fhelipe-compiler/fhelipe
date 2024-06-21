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

#ifndef FHELIPE_DIMENSION_BIT_H_
#define FHELIPE_DIMENSION_BIT_H_

#include <iostream>

#include "io_utils.h"

namespace fhelipe {

struct DimensionBit {
  int dimension;
  int bit_index;
  DimensionBit(int dimension = 0, int bit_index = 0)
      : dimension(dimension), bit_index(bit_index) {}
};

inline bool operator==(const DimensionBit& a, const DimensionBit& b) {
  return a.dimension == b.dimension && a.bit_index == b.bit_index;
}
inline bool operator<(const DimensionBit& a, const DimensionBit& b) {
  return a.dimension < b.dimension ||
         (a.dimension == b.dimension && a.bit_index < b.bit_index);
}
inline bool operator<=(const DimensionBit& a, const DimensionBit& b) {
  return a < b || a == b;
}
inline bool operator!=(const DimensionBit& a, const DimensionBit& b) {
  return !(a == b);
}
inline bool operator>(const DimensionBit& a, const DimensionBit& b) {
  return b < a;
}
inline bool operator>=(const DimensionBit& a, const DimensionBit& b) {
  return b <= a;
}

template <>
inline void WriteStream<DimensionBit>(std::ostream& stream,
                                      const DimensionBit& db) {
  stream << db.dimension << " " << db.bit_index;
}

}  // namespace fhelipe

#endif  // FHELIPE_DIMENSION_BIT_H_
