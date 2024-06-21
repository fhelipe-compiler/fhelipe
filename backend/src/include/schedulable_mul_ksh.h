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

#ifndef FHELIPE_SCHEDULABLE_MUL_KSH_H_
#define FHELIPE_SCHEDULABLE_MUL_KSH_H_

#include <glog/logging.h>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "ct_op_visitor.h"
#include "dimension_bit.h"
#include "io_c.h"
#include "io_spec.h"
#include "io_utils.h"
#include "tensor_index.h"
#include "tensor_layout.h"

namespace fhelipe {

class SchedulableMulKsh;

template <>
void WriteStream<SchedulableMulKsh>(std::ostream& stream,
                                    const SchedulableMulKsh& node);

class SchedulableMulKsh final : public CtOp {
 public:
  explicit SchedulableMulKsh(const Level& level) : CtOp({level, 0}) {}
  ~SchedulableMulKsh() final = default;
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<SchedulableMulKsh>(GetLevel());
  }
  const std::string& TypeName() const final { return StaticTypeName(); }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "SchedulableMulKsh";
    return type_name_;
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<SchedulableMulKsh>(stream, *this);
  }

 private:
  void WriteToStream(std::ostream& stream) const;

  static const std::string type_name_;
  static CtOpDerivedRegistrar<SchedulableMulKsh> reg_;
};

inline CtOpDerivedRegistrar<SchedulableMulKsh> SchedulableMulKsh::reg_{
    SchedulableMulKsh::StaticTypeName()};

template <>
inline void WriteStream<SchedulableMulKsh>(std::ostream& stream,
                                           const SchedulableMulKsh& node) {
  WriteStream<std::string>(stream, SchedulableMulKsh::StaticTypeName());
  stream << " ";
  WriteStream<Level>(stream, node.GetLevel());
}

template <>
inline SchedulableMulKsh ReadStreamWithoutTypeNamePrefix<SchedulableMulKsh>(
    std::istream& stream) {
  auto level = ReadStream<Level>(stream);
  return SchedulableMulKsh{level};
}

}  // namespace fhelipe

#endif  // FHELIPE_SCHEDULABLE_MUL_KSH_H_
