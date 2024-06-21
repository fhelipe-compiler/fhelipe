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

#ifndef FHELIPE_T_RESIZE_DIM_C_H_
#define FHELIPE_T_RESIZE_DIM_C_H_

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <vector>

#include "include/constants.h"
#include "include/laid_out_tensor.h"
#include "include/tensor_layout.h"
#include "shape.h"
#include "t_op.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TResizeDimC;

template <>
void WriteStream<TResizeDimC>(std::ostream& stream, const TResizeDimC& node);

class TResizeDimC final : public TOp {
 public:
  TResizeDimC(const TensorLayout& input_layout,
              const TensorLayout& output_layout);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& InputLayout() const { return input_layout_; }
  const TensorLayout& OutputLayout() const final { return output_layout_; }
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TResizeDimC>(*this);
  }
  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final;

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TResizeDimC>(stream, *this);
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TResizeDimC";
    return type_name_;
  }

 private:
  TensorLayout input_layout_;
  TensorLayout output_layout_;

  static TOpDerivedRegistrar<TResizeDimC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TResizeDimC> TResizeDimC::reg_{
    TResizeDimC::StaticTypeName()};

template <>
inline void WriteStream<TResizeDimC>(std::ostream& stream,
                                     const TResizeDimC& node) {
  WriteStream<std::string>(stream, TResizeDimC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
}

template <>
inline TResizeDimC ReadStreamWithoutTypeNamePrefix<TResizeDimC>(
    std::istream& stream) {
  auto input_layout = ReadStream<TensorLayout>(stream);
  auto output_layout = ReadStream<TensorLayout>(stream);
  return {input_layout, output_layout};
}
}  // namespace fhelipe

#endif  // FHELIPE_RESIZE_DIM_C_H_
