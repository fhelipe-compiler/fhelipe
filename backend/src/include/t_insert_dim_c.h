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

#ifndef FHELIPE_T_INSERT_DIM_C_H_
#define FHELIPE_T_INSERT_DIM_C_H_

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <vector>

#include "include/constants.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_utils.h"
#include "include/shape.h"
#include "include/tensor_layout.h"
#include "t_op.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

Shape GetOutputShapeTInsertDimC(const Shape& shape, int dim_to_insert);

class TInsertDimC;

template <>
void WriteStream<TInsertDimC>(std::ostream& stream, const TInsertDimC& node);

class TInsertDimC final : public TOp {
 public:
  TInsertDimC(const TensorLayout& layout, int dim_to_insert);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& InputLayout() const { return layout_; }
  const TensorLayout& OutputLayout() const;
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;
  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TInsertDimC>(*this);
  }
  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 0; }

  const std::string& TypeName() const { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TInsertDimC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TInsertDimC";
    return type_name_;
  }

  int DimensionToInsert() const { return dim_to_insert_; }

 private:
  TensorLayout layout_;
  TensorLayout output_layout_;
  int dim_to_insert_;

  static TOpDerivedRegistrar<TInsertDimC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TInsertDimC> TInsertDimC::reg_{
    TInsertDimC::StaticTypeName()};

template <>
inline void WriteStream<TInsertDimC>(std::ostream& stream,
                                     const TInsertDimC& node) {
  WriteStream<std::string>(stream, TInsertDimC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
  stream << " ";
  WriteStream<int>(stream, node.DimensionToInsert());
}

template <>
inline TInsertDimC ReadStreamWithoutTypeNamePrefix<TInsertDimC>(
    std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto dim_to_insert = ReadStream<int>(stream);
  return {tensor_layout, dim_to_insert};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_INSERT_DIM_C_H_
