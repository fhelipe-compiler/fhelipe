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

#ifndef FHELIPE_T_MUL_CSI_H_
#define FHELIPE_T_MUL_CSI_H_

#include <memory>
#include <ostream>
#include <vector>

#include "laid_out_tensor.h"
#include "plaintext.h"
#include "scaled_pt_val.h"
#include "shape.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TMulCSI;

template <>
void WriteStream<TMulCSI>(std::ostream& stream, const TMulCSI& node);

class TMulCSI final : public TOp {
 public:
  TMulCSI(const TensorLayout& layout, const ScaledPtVal& scalar);
  TOp::LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<TOp::LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }

  const ScaledPtVal& Scalar() const { return scalar_; }
  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TMulCSI>(*this);
  }
  LogScale AddedLogScale() const final { return scalar_.GetLogScale(); }
  int BackendMaskDepth() const final { return 0; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TMulCSI>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TAddCSI";
    return type_name_;
  }

  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

 private:
  TensorLayout layout_;
  ScaledPtVal scalar_;

  static TOpDerivedRegistrar<TMulCSI> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TMulCSI::TMulCSI(const TensorLayout& layout, const ScaledPtVal& scalar)
    : layout_(layout), scalar_(scalar) {}

inline TOpDerivedRegistrar<TMulCSI> TMulCSI::reg_{TMulCSI::StaticTypeName()};

template <>
inline void WriteStream<TMulCSI>(std::ostream& stream, const TMulCSI& node) {
  WriteStream<std::string>(stream, TMulCSI::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<ScaledPtVal>(stream, node.Scalar());
}  // namespace fhelipe

template <>
inline TMulCSI ReadStreamWithoutTypeNamePrefix<TMulCSI>(std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto scalar = ReadStream<ScaledPtVal>(stream);
  return {tensor_layout, scalar};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_MUL_CSI_H_
