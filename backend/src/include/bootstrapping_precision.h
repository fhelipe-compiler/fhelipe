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

#ifndef FHELIPE_BOOTSTRAPPING_PRECISION_H_
#define FHELIPE_BOOTSTRAPPING_PRECISION_H_

#include <glog/logging.h>

#include "framework/nominal.h"
#include "include/io_utils.h"

namespace fhelipe {

class BootstrappingPrecisionTypeIsolation {};

class BootstrappingPrecision
    : public fwk::Nominal<BootstrappingPrecisionTypeIsolation, int> {
 public:
  BootstrappingPrecision(int precision)
      : Nominal<BootstrappingPrecisionTypeIsolation, int>(precision) {
    CHECK(precision == 19 || precision == 26 || precision == 32);
  }
};

template <>
inline BootstrappingPrecision ReadStream<BootstrappingPrecision>(
    std::istream& stream) {
  return {ReadStream<int>(stream)};
}

template <>
inline void WriteStream<BootstrappingPrecision>(
    std::ostream& stream, const BootstrappingPrecision& bp) {
  WriteStream<int>(stream, bp.value());
}

}  // namespace fhelipe

#endif  // FHELIPE_BOOTSTRAPPING_PRECISION_H_
