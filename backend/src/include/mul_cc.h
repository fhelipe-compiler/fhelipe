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

#ifndef FHELIPE_MUL_CC_H_
#define FHELIPE_MUL_CC_H_

#include <string>

#include "ct_op.h"
#include "ct_op_visitor.h"
#include "io_utils.h"
#include "level_info.h"

namespace fhelipe {

template <>
void WriteStream<MulCC>(std::ostream& stream, const MulCC& node);

class CtOpVisitor;

class MulCC final : public CtOp {
 public:
  explicit MulCC(const LevelInfo& level_info);
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<MulCC>(GetLevelInfo());
  }

  const std::string& TypeName() const final { return StaticTypeName(); }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<MulCC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "MulCC";
    return type_name_;
  }

 private:
  static CtOpDerivedRegistrar<MulCC> reg_;
};

inline MulCC::MulCC(const LevelInfo& level_info) : CtOp(level_info) {}

inline CtOpDerivedRegistrar<MulCC> MulCC::reg_{MulCC::StaticTypeName()};

template <>
inline void WriteStream<MulCC>(std::ostream& stream, const MulCC& node) {
  WriteStream<std::string>(stream, MulCC::StaticTypeName());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline MulCC ReadStreamWithoutTypeNamePrefix<MulCC>(std::istream& stream) {
  return MulCC(ReadStream<LevelInfo>(stream));
}

}  // namespace fhelipe

#endif  // FHELIPE_MUL_CC_H_
