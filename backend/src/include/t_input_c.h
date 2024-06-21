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

#ifndef FHELIPE_T_INPUT_C_H_
#define FHELIPE_T_INPUT_C_H_

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "constants.h"
#include "include/ct_op.h"
#include "laid_out_tensor.h"
#include "shape.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {

class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TInputC;

template <>
void WriteStream<TInputC>(std::ostream& stream, const TInputC& node);

class TInputC final : public TOp {
 public:
  TInputC(const TensorLayout& layout, const std::string& name,
          LogScale log_scale);
  TOp::LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<TOp::LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }
  std::string Name() const { return name_; }
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TInputC>(*this);
  }
  TOp::LaidOutTensorCt CreateInputTensor(ct_program::CtProgram& ct_program,
                                         const LevelInfo& level_info) const;

  LogScale GetLogScale() const { return log_scale_; }
  LogScale AddedLogScale() const final { return 0; }

  int BackendMaskDepth() const final { return 0; }

  const std::string& TypeName() const { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TInputC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TInputC";
    return type_name_;
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

 private:
  TensorLayout layout_;
  std::string name_;
  LogScale log_scale_;

  static TOpDerivedRegistrar<TInputC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TInputC::TInputC(const TensorLayout& layout, const std::string& name,
                        LogScale log_scale)
    : layout_(layout), name_(name), log_scale_(log_scale) {}

inline TOpDerivedRegistrar<TInputC> TInputC::reg_{TInputC::StaticTypeName()};

template <>
inline void WriteStream<TInputC>(std::ostream& stream, const TInputC& node) {
  WriteStream<std::string>(stream, TInputC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<std::string>(stream, node.Name());
  stream << " ";
  WriteStream<LogScale>(stream, node.GetLogScale());
}

template <>
inline TInputC ReadStreamWithoutTypeNamePrefix<TInputC>(std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto tensor_name = ReadStream<std::string>(stream);
  auto log_scale = ReadStream<LogScale>(stream);
  return {tensor_layout, tensor_name, log_scale};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_INPUT_C_H_
