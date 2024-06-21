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

#ifndef FHELIPE_OUTPUT_C_H_
#define FHELIPE_OUTPUT_C_H_

#include <glog/logging.h>

#include <optional>
#include <ostream>
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

struct DimensionBit;

template <>
void WriteStream<OutputC>(std::ostream& stream, const OutputC& node);

class OutputC final : public IoC {
 public:
  OutputC(const LevelInfo& level_info, const IoSpec& io_spec);
  const IoSpec& GetIoSpec() const { return io_spec_; }
  const std::vector<std::optional<DimensionBit>>& Bits() const;
  OutputC(const OutputC& other)
      : IoC(other.GetLevelInfo()), io_spec_(other.GetIoSpec()) {}
  ~OutputC() final = default;
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<OutputC>(GetLevelInfo(), io_spec_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<OutputC>(stream, *this);
  }
  const std::string& TypeName() const final { return StaticTypeName(); }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "OutputC";
    return type_name_;
  }

 private:
  IoSpec io_spec_;

  static CtOpDerivedRegistrar<OutputC> reg_;
};

inline OutputC::OutputC(const LevelInfo& level_info, const IoSpec& io_spec)
    : IoC(level_info), io_spec_(io_spec) {}

inline CtOpDerivedRegistrar<OutputC> OutputC::reg_{OutputC::StaticTypeName()};

template <>
inline void WriteStream<OutputC>(std::ostream& stream, const OutputC& node) {
  WriteStream<std::string>(stream, OutputC::StaticTypeName());
  stream << " ";
  WriteStream(stream, node.GetIoSpec());
  stream << " ";
  WriteStream(stream, node.GetLevelInfo());
}

template <>
inline OutputC ReadStreamWithoutTypeNamePrefix<OutputC>(std::istream& stream) {
  auto io_spec = ReadStream<IoSpec>(stream);
  auto level_info = ReadStream<LevelInfo>(stream);
  return {level_info, io_spec};
}

}  // namespace fhelipe

#endif  // FHELIPE_OUTPUT_C_H_
