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

#ifndef FHELIPE_RESCALE_C_H_
#define FHELIPE_RESCALE_C_H_

#include "ct_op.h"
#include "ct_op_visitor.h"
#include "level_info.h"

namespace fhelipe {

class RescaleC;

template <>
void WriteStream<RescaleC>(std::ostream& stream, const RescaleC& node);

class RescaleC final : public CtOp {
 public:
  explicit RescaleC(const LevelInfo& level_info);
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<RescaleC>(GetLevelInfo());
  }
  const std::string& TypeName() const final { return StaticTypeName(); }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<RescaleC>(stream, *this);
  }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "RescaleC";
    return type_name_;
  }

 private:
  static CtOpDerivedRegistrar<RescaleC> reg_;
};

inline RescaleC::RescaleC(const LevelInfo& level_info) : CtOp(level_info) {}

inline CtOpDerivedRegistrar<RescaleC> RescaleC::reg_{
    RescaleC::StaticTypeName()};

template <>
inline void WriteStream<RescaleC>(std::ostream& stream, const RescaleC& node) {
  WriteStream<std::string>(stream, RescaleC::StaticTypeName());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline RescaleC ReadStreamWithoutTypeNamePrefix<RescaleC>(
    std::istream& stream) {
  auto level_info = ReadStream<LevelInfo>(stream);
  return RescaleC{level_info};
}

}  // namespace fhelipe

#endif  // FHELIPE_RESCALE _C_H_
