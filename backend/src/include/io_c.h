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

#ifndef FHELIPE_IO_C_H_
#define FHELIPE_IO_C_H_

#include <memory>

#include "ct_op.h"
#include "io_spec.h"

namespace fhelipe {
class IoC : public CtOp {
 public:
  explicit IoC(const LevelInfo& level_info) : CtOp(level_info) {}
  virtual const IoSpec& GetIoSpec() const = 0;

  virtual ~IoC() = default;
};

inline bool operator==(const IoC& lhs, const IoC& rhs) {
  return lhs.GetIoSpec() == rhs.GetIoSpec();
}

}  // namespace fhelipe

#endif  // FHELIPE_IO_C_H_
