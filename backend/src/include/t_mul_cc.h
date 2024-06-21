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

#ifndef FHELIPE_T_MUL_CC_H_
#define FHELIPE_T_MUL_CC_H_

#include <memory>
#include <ostream>
#include <vector>

#include "include/constants.h"
#include "laid_out_tensor.h"
#include "shape.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {
class CtOp;

namespace ct_dag {
class CtDag;
}  // namespace ct_dag

class TMulCC;

template <>
void WriteStream<TMulCC>(std::ostream& stream, const TMulCC& node);

class TMulCC final : public TOp {
 public:
  explicit TMulCC(const TensorLayout& layout);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }
  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TMulCC>(*this);
  }
  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 0; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TMulCC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TMulCC";
    return type_name_;
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

 private:
  TensorLayout layout_;
  bool EqualTo(const TOp& other) const final;

  static TOpDerivedRegistrar<TMulCC> reg_;
};

inline TOpDerivedRegistrar<TMulCC> TMulCC::reg_{TMulCC::StaticTypeName()};

template <>
inline void WriteStream<TMulCC>(std::ostream& stream, const TMulCC& node) {
  WriteStream<std::string>(stream, TMulCC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
}

template <>
inline TMulCC ReadStreamWithoutTypeNamePrefix<TMulCC>(std::istream& stream) {
  return TMulCC(ReadStream<TensorLayout>(stream));
}

}  // namespace fhelipe

#endif  // FHELIPE_T_MUL_CC_H_
