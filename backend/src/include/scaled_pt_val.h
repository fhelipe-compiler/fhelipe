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

#ifndef FHELIPE_SCALED_PT_VAL_H_
#define FHELIPE_SCALED_PT_VAL_H_

#include "io_utils.h"
#include "log_scale.h"
#include "plaintext.h"

namespace fhelipe {

class ScaledPtVal {
 public:
  ScaledPtVal(LogScale log_scale, PtVal value)
      : log_scale_(log_scale), value_(value) {}

  LogScale GetLogScale() const { return log_scale_; }
  PtVal value() const { return value_; }

 private:
  LogScale log_scale_;
  PtVal value_;
};

inline bool operator==(const ScaledPtVal& lhs, const ScaledPtVal& rhs) {
  return lhs.value() == rhs.value() && lhs.GetLogScale() == rhs.GetLogScale();
}

template <>
inline void WriteStream<ScaledPtVal>(std::ostream& stream,
                                     const ScaledPtVal& value) {
  WriteStream<LogScale>(stream, value.GetLogScale());
  stream << " ";
  WriteStream<PtVal>(stream, value.value());
}

template <>
inline ScaledPtVal ReadStream<ScaledPtVal>(std::istream& stream) {
  auto log_scale = ReadStream<LogScale>(stream);
  auto value = ReadStream<PtVal>(stream);
  return {log_scale, value};
}

}  // namespace fhelipe

#endif  // FHELIPE_SCALED_PT_VAL_H_
