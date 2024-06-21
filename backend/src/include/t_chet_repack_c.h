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

#ifndef FHELIPE_T_CHET_REPACK_H_
#define FHELIPE_T_CHET_REPACK_H_

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

class TChetRepackC;

TensorLayout ChetRepackedLayout(const LogChunkSize& log_chunk_size,
                                const Shape& shape);
template <>
void WriteStream<TChetRepackC>(std::ostream& stream, const TChetRepackC& node);

class TChetRepackC final : public TOp {
 public:
  explicit TChetRepackC(const TensorLayout& layout);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TChetRepackC>(*this);
  }
  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final {
    if (input_layout_ == output_layout_) {
      return 0;
    }
    return 1;
  }

  const TensorLayout& InputLayout() const { return input_layout_; }
  const TensorLayout& OutputLayout() const final { return output_layout_; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TChetRepackC>(stream, *this);
  }
  // nsamar: I need to have this function instead of using type_name_ directly
  // because the initialization order of static variables across compilation
  // units is not defined.
  // Wrapping the variable in a function solves the problem.
  // https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TChetRepackC";
    return type_name_;
  }

 private:
  TensorLayout input_layout_;
  TensorLayout output_layout_;

  static TOpDerivedRegistrar<TChetRepackC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TChetRepackC> TChetRepackC::reg_{
    TChetRepackC::StaticTypeName()};

template <>
inline void WriteStream<TChetRepackC>(std::ostream& stream,
                                      const TChetRepackC& node) {
  WriteStream<std::string>(stream, node.TypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.InputLayout());
}

template <>
inline TChetRepackC ReadStreamWithoutTypeNamePrefix<TChetRepackC>(
    std::istream& stream) {
  return TChetRepackC{ReadStream<TensorLayout>(stream)};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_CHET_REPACK_H_
