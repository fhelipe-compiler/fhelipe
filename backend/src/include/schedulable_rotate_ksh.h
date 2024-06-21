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

#ifndef FHELIPE_SCHEDULABLE_ROTATE_KSH_H_
#define FHELIPE_SCHEDULABLE_ROTATE_KSH_H_

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

class SchedulableRotateKsh;

template <>
void WriteStream<SchedulableRotateKsh>(std::ostream& stream,
                                       const SchedulableRotateKsh& node);

class SchedulableRotateKsh final : public CtOp {
 public:
  explicit SchedulableRotateKsh(const Level& level, int rotate_by)
      : CtOp({level, 0}), rotate_by_(rotate_by) {}
  ~SchedulableRotateKsh() final = default;
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<SchedulableRotateKsh>(GetLevel(), RotateBy());
  }
  int RotateBy() const { return rotate_by_; }
  const std::string& TypeName() const final { return StaticTypeName(); }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "SchedulableRotateKsh";
    return type_name_;
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<SchedulableRotateKsh>(stream, *this);
  }

 private:
  int rotate_by_;
  void WriteToStream(std::ostream& stream) const;

  static const std::string type_name_;
  static CtOpDerivedRegistrar<SchedulableRotateKsh> reg_;
};

inline CtOpDerivedRegistrar<SchedulableRotateKsh> SchedulableRotateKsh::reg_{
    SchedulableRotateKsh::StaticTypeName()};

template <>
inline void WriteStream<SchedulableRotateKsh>(
    std::ostream& stream, const SchedulableRotateKsh& node) {
  WriteStream<std::string>(stream, SchedulableRotateKsh::StaticTypeName());
  stream << " ";
  WriteStream<Level>(stream, node.GetLevel());
  stream << " ";
  WriteStream<int>(stream, node.RotateBy());
}

template <>
inline SchedulableRotateKsh
ReadStreamWithoutTypeNamePrefix<SchedulableRotateKsh>(std::istream& stream) {
  auto level = ReadStream<Level>(stream);
  auto rotate_by = ReadStream<int>(stream);
  return SchedulableRotateKsh{level, rotate_by};
}

}  // namespace fhelipe

#endif  // FHELIPE_SCHEDULABLE_ROTATE_KSH_H_
