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

#ifndef CHERITON_NOMINAL_H_
#define CHERITON_NOMINAL_H_

namespace fwk {

template <class UnitType, class BaseType>
class Nominal {
 public:
  explicit Nominal(BaseType v = BaseType()) : value_(v) {}
  bool operator==(const Nominal<UnitType, BaseType>& v) const {
    return value() == v.value();
  }
  bool operator!=(const Nominal<UnitType, BaseType>& v) const {
    return value() != v.value();
  }

  explicit operator bool() const { return value_; }

  BaseType value() const { return value_; }

 protected:
  void valueIs(BaseType v) { value_ = v; }
  BaseType& valueRef() { return value_; }

 private:
  BaseType value_;
};

}  // namespace fwk

#endif  // CHERITON_NOMINAL_H_
