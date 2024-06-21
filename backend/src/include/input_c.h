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

#ifndef FHELIPE_INPUT_C_H_
#define FHELIPE_INPUT_C_H_

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

template <>
void WriteStream<InputC>(std::ostream& stream, const InputC& node);

class InputC final : public IoC {
 public:
  InputC(const LevelInfo& level_info, const IoSpec& io_spec)
      : IoC(level_info), io_spec_(io_spec) {}
  const IoSpec& GetIoSpec() const { return io_spec_; }
  InputC(const InputC& other)
      : IoC(other.GetLevelInfo()), io_spec_(other.GetIoSpec()) {}
  ~InputC() final = default;
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<InputC>(GetLevelInfo(), io_spec_);
  }
  const std::string& TypeName() const final { return StaticTypeName(); }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "InputC";
    return type_name_;
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<InputC>(stream, *this);
  }

 private:
  IoSpec io_spec_;
  void WriteToStream(std::ostream& stream) const;

  static const std::string type_name_;
  static CtOpDerivedRegistrar<InputC> reg_;
};

inline CtOpDerivedRegistrar<InputC> InputC::reg_{InputC::StaticTypeName()};

template <>
inline void WriteStream<InputC>(std::ostream& stream, const InputC& node) {
  WriteStream<std::string>(stream, InputC::StaticTypeName());
  stream << " ";
  WriteStream<IoSpec>(stream, node.GetIoSpec());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline InputC ReadStreamWithoutTypeNamePrefix<InputC>(std::istream& stream) {
  auto io_spec = ReadStream<IoSpec>(stream);
  auto level_info = ReadStream<LevelInfo>(stream);
  return {level_info, io_spec};
}

}  // namespace fhelipe

#endif  // FHELIPE_INPUT_C_H_
