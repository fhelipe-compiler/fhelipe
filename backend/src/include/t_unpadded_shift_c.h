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

#ifndef FHELIPE_T_UNPADDED_SHIFT_C_H_
#define FHELIPE_T_UNPADDED_SHIFT_C_H_

#include <memory>
#include <ostream>
#include <vector>

#include "include/constants.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_generator.h"
#include "laid_out_tensor.h"
#include "t_op.h"
#include "tensor_index.h"

namespace fhelipe {

class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TUnpaddedShiftC;

template <>
void WriteStream<TUnpaddedShiftC>(std::ostream& stream,
                                  const TUnpaddedShiftC& node);

class TUnpaddedShiftC final : public TOp {
 public:
  TUnpaddedShiftC(const TensorLayout& layout, const DiffTensorIndex& rotate_by);
  virtual ~TUnpaddedShiftC() = default;
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;

  const TensorLayout& OutputLayout() const final { return layout_; }
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TUnpaddedShiftC>(*this);
  }
  const DiffTensorIndex& RotateBy() const { return rotate_by_; }
  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final;

  const DiffTensorIndex& GetDiffTensorIndex() const { return rotate_by_; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TUnpaddedShiftC>(stream, *this);
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TUnpaddedShiftC";
    return type_name_;
  }

 private:
  TensorLayout layout_;
  DiffTensorIndex rotate_by_;
  std::vector<TranslationMask> translation_masks_;

  static TOpDerivedRegistrar<TUnpaddedShiftC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TUnpaddedShiftC> TUnpaddedShiftC::reg_{
    TUnpaddedShiftC::StaticTypeName()};

template <>
inline void WriteStream<TUnpaddedShiftC>(std::ostream& stream,
                                         const TUnpaddedShiftC& node) {
  WriteStream<std::string>(stream, node.TypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<DiffTensorIndex>(stream, node.GetDiffTensorIndex());
}

template <>
inline TUnpaddedShiftC ReadStreamWithoutTypeNamePrefix<TUnpaddedShiftC>(
    std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto rotate_by = ReadStream<DiffTensorIndex>(stream);
  return {tensor_layout, rotate_by};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_UNPADDED_SHIFT_C_H_
