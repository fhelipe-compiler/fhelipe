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

#ifndef FHELIPE_T_OUTPUT_C_H_
#define FHELIPE_T_OUTPUT_C_H_

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "constants.h"
#include "laid_out_tensor.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {

class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TOutputC;

template <>
void WriteStream<TOutputC>(std::ostream& stream, const TOutputC& node);

class TOutputC final : public TOp {
 public:
  TOutputC(const TensorLayout& layout, const std::string& name);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& in_tensor) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }
  std::string Name() const { return name_; }
  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TOutputC>(*this);
  }

  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 0; }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TOutputC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TOutputC";
    return type_name_;
  }

 private:
  TensorLayout layout_;
  std::string name_;

  static TOpDerivedRegistrar<TOutputC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOutputC::TOutputC(const TensorLayout& layout, const std::string& name)
    : layout_(layout), name_(name) {}

inline TOpDerivedRegistrar<TOutputC> TOutputC::reg_{TOutputC::StaticTypeName()};

template <>
inline void WriteStream<TOutputC>(std::ostream& stream, const TOutputC& node) {
  WriteStream<std::string>(stream, TOutputC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<std::string>(stream, node.Name());
}

template <>
inline TOutputC ReadStreamWithoutTypeNamePrefix<TOutputC>(
    std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto tensor_name = ReadStream<std::string>(stream);
  return {tensor_layout, tensor_name};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_OUTPUT_C_H_
