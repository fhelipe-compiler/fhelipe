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

#ifndef CHERITON_INTERVAL_H_
#define CHERITON_INTERVAL_H_

#include <glog/logging.h>

#include "ordinal.h"

namespace fwk {

template <class UnitType, class BaseType, class DiffType>
class Interval : public Ordinal<UnitType, BaseType> {
 public:
  explicit Interval(BaseType v = BaseType()) : Ordinal<UnitType, BaseType>(v) {}
  DiffType operator-(const Interval<UnitType, BaseType, DiffType>& v) const {
    return DiffType(value() - v.value());
  }

 protected:
  using Nominal<UnitType, BaseType>::value;
  using Nominal<UnitType, BaseType>::valueRef;
};

template <class UnitType, class BaseType, class DiffType>
inline Interval<UnitType, BaseType, DiffType> operator*(
    int lhs, Interval<UnitType, BaseType, DiffType> rhs) {
  CHECK(lhs >= 0);
  Interval<UnitType, BaseType, DiffType> result;
  for (int i = 0; i < lhs; ++i) {
    result += rhs;
  }
  return result;
}

}  // namespace fwk

#endif  // CHERITON_INTERVAL_H_
