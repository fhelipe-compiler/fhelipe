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

#ifndef FHELIPE_T_DROP_DIM_C_H_
#define FHELIPE_T_DROP_DIM_C_H_

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <vector>

#include "include/constants.h"
#include "include/laid_out_tensor.h"
#include "include/shape.h"
#include "include/tensor_layout.h"
#include "t_op.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

Shape GetOutputShapeTDropDimC(const Shape& shape, int dim_to_drop);

class TDropDimC;

template <>
void WriteStream<TDropDimC>(std::ostream& stream, const TDropDimC& node);

class TDropDimC final : public TOp {
 public:
  TDropDimC(const TensorLayout& layout, int dim_to_drop);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& InputLayout() const { return layout_; }
  const TensorLayout& OutputLayout() const { return output_layout_; }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TDropDimC>(*this);
  }

  const std::string& TypeName() const { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TDropDimC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TDropDimC";
    return type_name_;
  }

  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 0; }
  int DimensionToDrop() const { return dim_to_drop_; }

 private:
  TensorLayout layout_;
  TensorLayout output_layout_;
  int dim_to_drop_;

  static TOpDerivedRegistrar<TDropDimC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TDropDimC> TDropDimC::reg_{
    TDropDimC::StaticTypeName()};

template <>
inline void WriteStream<TDropDimC>(std::ostream& stream,
                                   const TDropDimC& node) {
  WriteStream<std::string>(stream, TDropDimC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
  stream << " ";
  WriteStream<int>(stream, node.DimensionToDrop());
}

template <>
inline TDropDimC ReadStreamWithoutTypeNamePrefix<TDropDimC>(
    std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto dim_to_drop = ReadStream<int>(stream);
  return {tensor_layout, dim_to_drop};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_DROP_DIM_C_H_
