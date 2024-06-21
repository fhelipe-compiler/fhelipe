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

#ifndef FHELIPE_T_STRIDE_C_H_
#define FHELIPE_T_STRIDE_C_H_

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <vector>

#include "include/constants.h"
#include "include/io_utils.h"
#include "include/laid_out_tensor.h"
#include "include/shape.h"
#include "include/tensor_layout.h"
#include "t_op.h"
#include "utils.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class Stride {
 public:
  Stride(int stride);
  int value() const { return stride_; }

 private:
  int stride_;
};

template <>
inline void WriteStream<Stride>(std::ostream& stream, const Stride& stride) {
  WriteStream<int>(stream, stride.value());
}

template <>
inline Stride ReadStream<Stride>(std::istream& stream) {
  return ReadStream<int>(stream);
}

Shape GetOutputShapeTStrideC(const Shape& shape,
                             const std::vector<Stride>& strides);

class TStrideC;

template <>
void WriteStream<TStrideC>(std::ostream& stream, const TStrideC& node);

class TStrideC final : public TOp {
 public:
  TStrideC(const TensorLayout& input_layout, const TensorLayout& output_layout,
           const std::vector<Stride>& strides);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 1; }
  const TensorLayout& InputLayout() const { return input_layout_; }
  const TensorLayout& OutputLayout() const final { return output_layout_; }
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TStrideC>(*this);
  }

  const std::vector<Stride>& Strides() const { return strides_; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TStrideC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TStrideC";
    return type_name_;
  }

 private:
  TensorLayout input_layout_;
  TensorLayout output_layout_;
  std::vector<Stride> strides_;

  static TOpDerivedRegistrar<TStrideC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline Stride::Stride(int stride) : stride_(stride) {
  CHECK(IsPowerOfTwo(stride_));
}

inline TOpDerivedRegistrar<TStrideC> TStrideC::reg_{TStrideC::StaticTypeName()};

template <>
inline void WriteStream<TStrideC>(std::ostream& stream, const TStrideC& node) {
  WriteStream<std::string>(stream, TStrideC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream(stream, node.Strides());
}

template <>
inline TStrideC ReadStreamWithoutTypeNamePrefix<TStrideC>(
    std::istream& stream) {
  auto input_layout = ReadStream<TensorLayout>(stream);
  auto output_layout = ReadStream<TensorLayout>(stream);
  auto strides = ReadStream<std::vector<Stride>>(stream);
  return {input_layout, output_layout, strides};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_STRIDE_C_H_
