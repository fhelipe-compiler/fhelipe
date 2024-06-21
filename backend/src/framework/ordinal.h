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

#ifndef CHERITON_ORDINAL_H_
#define CHERITON_ORDINAL_H_

#include "nominal.h"

namespace fwk {

template <class UnitType, class BaseType>
class Ordinal : public Nominal<UnitType, BaseType> {
 public:
  explicit Ordinal(BaseType v = BaseType()) : Nominal<UnitType, BaseType>(v) {}
  bool operator<(const Ordinal<UnitType, BaseType> v) const {
    return value() < v.value();
  }
  bool operator<=(const Ordinal<UnitType, BaseType> v) const {
    return value() <= v.value();
  }
  bool operator>(const Ordinal<UnitType, BaseType> v) const {
    return value() > v.value();
  }
  bool operator>=(const Ordinal<UnitType, BaseType> v) const {
    return value() >= v.value();
  }

 protected:
  using Nominal<UnitType, BaseType>::value;
  using Nominal<UnitType, BaseType>::valueRef;
};

}  // namespace fwk

#endif  // CHERITON_ORDINAL_H_
