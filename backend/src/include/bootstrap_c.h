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

#ifndef FHELIPE_BOOTSTRAP_C_H_
#define FHELIPE_BOOTSTRAP_C_H_

#include <string>

#include "ct_op.h"
#include "ct_op_visitor.h"

namespace fhelipe {

template <>
void WriteStream<BootstrapC>(std::ostream& stream, const BootstrapC& node);

class BootstrapC final : public CtOp {
 public:
  explicit BootstrapC(const LevelInfo& level_info) : CtOp(level_info) {}
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<BootstrapC>(GetLevelInfo());
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<BootstrapC>(stream, *this);
  }

  const std::string& TypeName() const final { return StaticTypeName(); }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "BootstrapC";
    return type_name_;
  }

 private:
  void WriteToStream(std::ostream& stream) const;

  static const std::string type_name_;
  static CtOpDerivedRegistrar<BootstrapC> reg_;
};

inline CtOpDerivedRegistrar<BootstrapC> BootstrapC::reg_{
    BootstrapC::StaticTypeName()};

template <>
inline void WriteStream<BootstrapC>(std::ostream& stream,
                                    const BootstrapC& node) {
  WriteStream<std::string>(stream, BootstrapC::StaticTypeName());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline BootstrapC ReadStreamWithoutTypeNamePrefix<BootstrapC>(
    std::istream& stream) {
  auto level_info = ReadStream<LevelInfo>(stream);
  return BootstrapC(level_info);
}

}  // namespace fhelipe

#endif  // FHELIPE_BOOTSTRAP_C_H_
