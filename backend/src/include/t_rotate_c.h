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

#ifndef FHELIPE_T_ROTATE_C_H_
#define FHELIPE_T_ROTATE_C_H_

#include <memory>
#include <ostream>
#include <vector>

#include "constants.h"
#include "laid_out_tensor.h"
#include "t_op.h"
#include "tensor_index.h"
#include "tensor_layout.h"

namespace fhelipe {
namespace ct_program {
class CtProgram;
}  // namespace ct_program

class CtOp;

class TRotateC;

template <>
void WriteStream<TRotateC>(std::ostream& stream, const TRotateC& node);

class TRotateC final : public TOp {
 public:
  TRotateC(const TensorLayout& layout, int rotate_by);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_progrm,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }
  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TRotateC>(*this);
  }

  const std::string& TypeName() const { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TRotateC>(stream, *this);
  }

  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 0; }
  int RotateBy() const { return rotate_by_; }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TRotateC";
    return type_name_;
  }

 private:
  TensorLayout layout_;
  int rotate_by_;

  static TOpDerivedRegistrar<TRotateC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TRotateC> TRotateC::reg_{TRotateC::StaticTypeName()};

template <>
inline void WriteStream<TRotateC>(std::ostream& stream, const TRotateC& node) {
  WriteStream<std::string>(stream, TRotateC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<int>(stream, node.RotateBy());
}

template <>
inline TRotateC ReadStreamWithoutTypeNamePrefix<TRotateC>(
    std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto rotate_by = ReadStream<int>(stream);
  return {tensor_layout, rotate_by};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_ROTATE_C_H_
