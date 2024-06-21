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

#ifndef FHELIPE_T_LAYOUT_CONVERSION_C_H_
#define FHELIPE_T_LAYOUT_CONVERSION_C_H_

#include <memory>
#include <ostream>
#include <vector>

#include "include/constants.h"
#include "laid_out_tensor.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {

class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TLayoutConversionC;

template <>
void WriteStream<TLayoutConversionC>(std::ostream& stream,
                                     const TLayoutConversionC& node);

class TLayoutConversionC final : public TOp {
 public:
  TLayoutConversionC(const TensorLayout& input_layout,
                     const TensorLayout& output_layout);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;

  const TensorLayout& InputLayout() const { return input_layout_; }
  const TensorLayout& OutputLayout() const final { return output_layout_; }
  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TLayoutConversionC>(*this);
  }
  int BackendMaskDepth() const final { return 1; }
  LogScale AddedLogScale() const final { return 0; }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TLayoutConversionC>(stream, *this);
  }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TLayoutConversionC";
    return type_name_;
  }

 private:
  TensorLayout input_layout_;
  TensorLayout output_layout_;

  static TOpDerivedRegistrar<TLayoutConversionC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TLayoutConversionC::TLayoutConversionC(const TensorLayout& input_layout,
                                              const TensorLayout& output_layout)
    : input_layout_(input_layout), output_layout_(output_layout) {}

inline TOpDerivedRegistrar<TLayoutConversionC> TLayoutConversionC::reg_{
    TLayoutConversionC::StaticTypeName()};

template <>
inline void WriteStream<TLayoutConversionC>(std::ostream& stream,
                                            const TLayoutConversionC& node) {
  WriteStream<std::string>(stream, TLayoutConversionC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
}

template <>
inline TLayoutConversionC ReadStreamWithoutTypeNamePrefix<TLayoutConversionC>(
    std::istream& stream) {
  auto input_layout = ReadStream<TensorLayout>(stream);
  auto output_layout = ReadStream<TensorLayout>(stream);
  return {input_layout, output_layout};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_LAYOUT_CONVERSION_C_H_
