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

#ifndef FHELIPE_T_CYCLIC_SHIFT_C_H_
#define FHELIPE_T_CYCLIC_SHIFT_C_H_

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

TensorIndex CyclicallyShiftedTensorIndex(const TensorIndex& ti,
                                         const TensorIndex& rotate_by);

class CtOp;

class TCyclicShiftC;

template <>
void WriteStream<TCyclicShiftC>(std::ostream& stream,
                                const TCyclicShiftC& node);

class TCyclicShiftC final : public TOp {
 public:
  TCyclicShiftC(const TensorLayout& layout, const DiffTensorIndex& rotate_by);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_progrm,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }
  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TCyclicShiftC>(*this);
  }

  const std::string& TypeName() const { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TCyclicShiftC>(stream, *this);
  }

  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 1; }
  const DiffTensorIndex& GetDiffTensorIndex() const { return rotate_by_; }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TCyclicShiftC";
    return type_name_;
  }

 private:
  TensorLayout layout_;
  DiffTensorIndex rotate_by_;

  static TOpDerivedRegistrar<TCyclicShiftC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TCyclicShiftC> TCyclicShiftC::reg_{
    TCyclicShiftC::StaticTypeName()};

template <>
inline void WriteStream<TCyclicShiftC>(std::ostream& stream,
                                       const TCyclicShiftC& node) {
  WriteStream<std::string>(stream, TCyclicShiftC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<DiffTensorIndex>(stream, node.GetDiffTensorIndex());
}

template <>
inline TCyclicShiftC ReadStreamWithoutTypeNamePrefix<TCyclicShiftC>(
    std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto diff_tensor_index = ReadStream<DiffTensorIndex>(stream);
  return {tensor_layout, diff_tensor_index};
}
inline TCyclicShiftC::TCyclicShiftC(const TensorLayout& layout,
                                    const DiffTensorIndex& rotate_by)
    : layout_(layout), rotate_by_(rotate_by) {}

}  // namespace fhelipe

#endif  // FHELIPE_T_CYCLIC_SHIFT_C_H_
