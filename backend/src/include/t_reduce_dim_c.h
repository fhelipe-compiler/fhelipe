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

#ifndef FHELIPE_T_REDUCE_DIM_C_H_
#define FHELIPE_T_REDUCE_DIM_C_H_

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <vector>

#include "constants.h"
#include "laid_out_tensor.h"
#include "shape.h"
#include "t_op.h"
#include "t_resize_dim_c.h"
#include "tensor_index.h"
#include "tensor_layout.h"
#include "utils.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

Shape GetOutputShapeTReduceDimC(Shape shape, int dimension);

class TReduceDimC;

template <>
void WriteStream<TReduceDimC>(std::ostream& stream, const TReduceDimC& node);

class TReduceDimC final : public TOp {
 public:
  TReduceDimC(const TensorLayout& input_layout,
              const TensorLayout& output_layout, int dimension);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& InputLayout() const { return input_layout_; }
  const TensorLayout& OutputLayout() const final { return output_layout_; }
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TReduceDimC>(*this);
  }

  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final {
    return TResizeDimC(input_layout_, output_layout_).BackendMaskDepth();
  }

  const std::string& TypeName() const final { return StaticTypeName(); }

  int DimensionToReduce() const { return dimension_; }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TReduceDimC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TReduceDimC";
    return type_name_;
  }

 private:
  TensorLayout input_layout_;
  TensorLayout output_layout_;
  int dimension_;

  static TOpDerivedRegistrar<TReduceDimC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TReduceDimC> TReduceDimC::reg_{
    TReduceDimC::StaticTypeName()};

template <>
inline void WriteStream<TReduceDimC>(std::ostream& stream,
                                     const TReduceDimC& node) {
  WriteStream<std::string>(stream, TReduceDimC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<int>(stream, node.DimensionToReduce());
}

template <>
inline TReduceDimC ReadStreamWithoutTypeNamePrefix<TReduceDimC>(
    std::istream& stream) {
  auto input_layout = ReadStream<TensorLayout>(stream);
  auto output_layout = ReadStream<TensorLayout>(stream);
  auto dimension = ReadStream<int>(stream);
  return {input_layout, output_layout, dimension};
}
}  // namespace fhelipe

#endif  // FHELIPE_T_REDUCE_DIM_C_H_
