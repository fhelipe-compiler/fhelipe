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

#ifndef FHELIPE_SCALED_T_OP_H_
#define FHELIPE_SCALED_T_OP_H_

#include <glog/logging.h>

#include <memory>

#include "log_scale.h"
#include "t_op.h"

namespace fhelipe {

class ScaledTOp {
 public:
  ScaledTOp(std::unique_ptr<TOp>&& t_op, LogScale log_scale)
      : t_op_(std::move(t_op)), log_scale_(log_scale) {}

  LogScale LogScale() const { return log_scale_; }
  const TOp& GetTOp() const { return *t_op_; }
  std::unique_ptr<ScaledTOp> CloneUniq() const {
    return std::make_unique<ScaledTOp>(t_op_->CloneUniq(), log_scale_);
  }

 private:
  std::unique_ptr<TOp> t_op_;
  class LogScale log_scale_;
};

template <>
inline void WriteStream<ScaledTOp>(std::ostream& stream,
                                   const ScaledTOp& t_op) {
  WriteStream(stream, t_op.GetTOp());
  stream << " ";
  WriteStream(stream, t_op.LogScale());
}

template <>
inline ScaledTOp ReadStream<ScaledTOp>(std::istream& stream) {
  auto t_op = TOp::CreateInstance(stream);
  auto log_scale = ReadStream<LogScale>(stream);
  return {std::move(t_op), log_scale};
}

inline std::ostream& operator<<(std::ostream& stream, const ScaledTOp& t_op) {
  WriteStream<ScaledTOp>(stream, t_op);
  return stream;
}

}  // namespace fhelipe

#endif  // FHELIPE_SCALED_T_OP_H_
