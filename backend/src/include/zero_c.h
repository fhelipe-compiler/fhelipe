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

#ifndef FHELIPE_ZERO_C_H_
#define FHELIPE_ZERO_C_H_

#include "ct_op.h"
#include "io_utils.h"
#include "level_info.h"

namespace fhelipe {

class ZeroC;

template <>
void WriteStream<ZeroC>(std::ostream& stream, const ZeroC& node);

class ZeroC final : public CtOp {
 public:
  explicit ZeroC(const LevelInfo& level_info) : CtOp(level_info) {}
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<ZeroC>(GetLevelInfo());
  }
  const std::string& TypeName() const final { return StaticTypeName(); }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<ZeroC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "ZeroC";
    return type_name_;
  }

 private:
  static const std::string type_name_;
  static CtOpDerivedRegistrar<ZeroC> reg_;
};

inline CtOpDerivedRegistrar<ZeroC> ZeroC::reg_{ZeroC::StaticTypeName()};

template <>
inline void WriteStream<ZeroC>(std::ostream& stream, const ZeroC& node) {
  WriteStream<std::string>(stream, ZeroC::StaticTypeName());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline ZeroC ReadStreamWithoutTypeNamePrefix<ZeroC>(std::istream& stream) {
  return ZeroC(ReadStream<LevelInfo>(stream));
}

}  // namespace fhelipe

#endif  // FHELIPE_ZERO_C_H_
