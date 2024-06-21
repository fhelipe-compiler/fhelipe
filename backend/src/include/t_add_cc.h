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

#ifndef FHELIPE_T_ADD_CC_H_
#define FHELIPE_T_ADD_CC_H_

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "ct_program.h"
#include "laid_out_tensor.h"
#include "shape.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TAddCC;

template <>
void WriteStream<TAddCC>(std::ostream& stream, const TAddCC& node);

class TAddCC final : public TOp {
 public:
  explicit TAddCC(const TensorLayout& layout);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TAddCC>(*this);
  }
  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 0; }

  const TensorLayout& OutputLayout() const final { return layout_; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TAddCC>(stream, *this);
  }
  // nsamar: I need to have this function instead of using type_name_ directly
  // because the initialization order of static variables across compilation
  // units is not defined.
  // Wrapping the variable in a function solves the problem.
  // https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TAddCC";
    return type_name_;
  }

 private:
  TensorLayout layout_;

  static TOpDerivedRegistrar<TAddCC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TAddCC> reg_{TAddCC::StaticTypeName()};

template <>
inline void WriteStream<TAddCC>(std::ostream& stream, const TAddCC& node) {
  WriteStream<std::string>(stream, node.TypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
}

template <>
inline TAddCC ReadStreamWithoutTypeNamePrefix<TAddCC>(std::istream& stream) {
  return TAddCC(ReadStream<TensorLayout>(stream));
}

}  // namespace fhelipe

#endif  // FHELIPE_T_ADD_CC_H_
