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

#ifndef FHELIPE_T_REORDER_DIMS_C_H_
#define FHELIPE_T_REORDER_DIMS_C_H_

#include <algorithm>
#include <memory>
#include <ostream>
#include <vector>

#include "constants.h"
#include "io_utils.h"
#include "laid_out_tensor.h"
#include "shape.h"
#include "t_op.h"
#include "tensor_layout.h"
#include "utils.h"

namespace fhelipe {
class CtOp;

namespace ct_program {

class CtProgram;

}  // namespace ct_program

Shape GetOutputShapeTReorderDimsC(const Shape& shape,
                                  const std::vector<int>& dim_order);

class TReorderDimsC;

template <>
void WriteStream<TReorderDimsC>(std::ostream& stream,
                                const TReorderDimsC& node);

class TReorderDimsC final : public TOp {
 public:
  TReorderDimsC(const TensorLayout& input_layout,
                const TensorLayout& output_layout,
                const std::vector<int>& dim_order);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& InputLayout() const { return input_layout_; }
  const TensorLayout& OutputLayout() const final { return output_layout_; }
  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TReorderDimsC>(*this);
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;
  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final {
    // TODO(nsamar): Not sure this needs to be depth 1
    // Could get away with zero?
    return 1;
  }

  const std::vector<int>& DimensionOrder() const { return dim_order_; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TReorderDimsC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TReorderDimsC";
    return type_name_;
  }

 private:
  TensorLayout input_layout_;
  TensorLayout output_layout_;
  std::vector<int> dim_order_;

  static TOpDerivedRegistrar<TReorderDimsC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TReorderDimsC> TReorderDimsC::reg_{
    TReorderDimsC::StaticTypeName()};

template <>
inline void WriteStream<TReorderDimsC>(std::ostream& stream,
                                       const TReorderDimsC& node) {
  WriteStream<std::string>(stream, TReorderDimsC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream(stream, node.DimensionOrder());
}

template <>
inline TReorderDimsC ReadStreamWithoutTypeNamePrefix<TReorderDimsC>(
    std::istream& stream) {
  auto input_layout = ReadStream<TensorLayout>(stream);
  auto output_layout = ReadStream<TensorLayout>(stream);
  auto dim_order = ReadStream<std::vector<int>>(stream);
  return {input_layout, output_layout, dim_order};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_STRIDE_C_H_
