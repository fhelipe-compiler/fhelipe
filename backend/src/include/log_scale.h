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

#ifndef FHELIPE_LOG_SCALE_H_
#define FHELIPE_LOG_SCALE_H_

#include <cmath>

#include "framework/interval.h"
#include "include/io_utils.h"

namespace fhelipe {

class Scale {
 public:
  explicit Scale(double value) : value_(value) {}
  double value() const { return value_; }

 private:
  double value_;
};

class LogScaleTypeIsolation {};

class LogScale : public fwk::Interval<LogScaleTypeIsolation, int, int> {
 public:
  LogScale(Scale value)
      : Interval<LogScaleTypeIsolation, int, int>(std::log2(value.value())) {}
  LogScale(int value) : Interval<LogScaleTypeIsolation, int, int>(value) {
    CHECK(value >= 0 && value < 200);
  }

  using fwk::Interval<LogScaleTypeIsolation, int, int>::value;
};

template <>
inline LogScale ReadStream<LogScale>(std::istream& stream) {
  return {ReadStream<int>(stream)};
}

inline LogScale operator*(int lhs, LogScale log_scale) {
  return lhs * log_scale.value();
}

template <>
inline void WriteStream<LogScale>(std::ostream& stream,
                                  const LogScale& log_scale) {
  WriteStream<int>(stream, log_scale.value());
}

inline LogScale operator+(LogScale lhs, LogScale rhs) {
  return {lhs.value() + rhs.value()};
}

}  // namespace fhelipe

#endif  // FHELIPE_LOG_SCALE_H_
