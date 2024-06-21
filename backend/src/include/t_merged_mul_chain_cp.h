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

#ifndef FHELIPE_T_MERGED_MUL_CHAIN_CP_H_
#define FHELIPE_T_MERGED_MUL_CHAIN_CP_H_

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

namespace ct_dag {
class CtDag;
}  // namespace ct_dag

class TMergedMulChainCP;

template <>
void WriteStream<TMergedMulChainCP>(std::ostream& stream,
                                    const TMergedMulChainCP& node);

class TMergedMulChainCP final : public TOp {
 public:
  TMergedMulChainCP(const TensorLayout& input_layout,
                    const TensorLayout& output_layout);
  TOp::LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<TOp::LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return output_layout_; }
  const TensorLayout& InputLayout() const { return input_layout_; }
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TMergedMulChainCP>(*this);
  }

  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 0; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TMergedMulChainCP>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TMergedMulChainCP";
    return type_name_;
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

 private:
  TensorLayout input_layout_;
  TensorLayout output_layout_;

  static TOpDerivedRegistrar<TMergedMulChainCP> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TMergedMulChainCP> TMergedMulChainCP::reg_{
    TMergedMulChainCP::StaticTypeName()};

template <>
inline void WriteStream<TMergedMulChainCP>(std::ostream& stream,
                                           const TMergedMulChainCP& node) {
  WriteStream<std::string>(stream, TMergedMulChainCP::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
}

template <>
inline TMergedMulChainCP ReadStreamWithoutTypeNamePrefix<TMergedMulChainCP>(
    std::istream& stream) {
  auto input_layout = ReadStream<TensorLayout>(stream);
  auto output_layout = ReadStream<TensorLayout>(stream);
  return {input_layout, output_layout};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_MERGED_MUL_CHAIN_CP_H_
