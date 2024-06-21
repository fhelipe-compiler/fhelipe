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

#ifndef FHELIPE_ROTATE_C_H_
#define FHELIPE_ROTATE_C_H_

#include "ct_op.h"
#include "ct_op_visitor.h"
#include "level_info.h"

namespace fhelipe {

template <>
void WriteStream<RotateC>(std::ostream& stream, const RotateC& node);

class RotateC final : public CtOp {
 public:
  RotateC(const LevelInfo& level_info, int rotate_by);

  int RotateBy() const;

  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<RotateC>(GetLevelInfo(), rotate_by_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<RotateC>(stream, *this);
  }
  const std::string& TypeName() const final { return StaticTypeName(); }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "RotateC";
    return type_name_;
  }

 private:
  int rotate_by_;

  static CtOpDerivedRegistrar<RotateC> reg_;
};

inline RotateC::RotateC(const LevelInfo& level_info, int rotate_by)
    : CtOp(level_info), rotate_by_(rotate_by) {}

inline int RotateC::RotateBy() const { return rotate_by_; }

inline CtOpDerivedRegistrar<RotateC> RotateC::reg_{RotateC::StaticTypeName()};

template <>
inline void WriteStream<RotateC>(std::ostream& stream, const RotateC& node) {
  WriteStream<std::string>(stream, RotateC::StaticTypeName());
  stream << " ";
  WriteStream<int>(stream, node.RotateBy());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline RotateC ReadStreamWithoutTypeNamePrefix<RotateC>(std::istream& stream) {
  auto rotate_by = ReadStream<int>(stream);
  auto level_info = ReadStream<LevelInfo>(stream);
  return {level_info, rotate_by};
}

}  // namespace fhelipe

#endif  // FHELIPE_ROTATE_C_H_
