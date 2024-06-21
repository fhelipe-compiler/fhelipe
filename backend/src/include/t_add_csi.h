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

#ifndef FHELIPE_T_ADD_CSI_H_
#define FHELIPE_T_ADD_CSI_H_

#include <memory>
#include <ostream>
#include <vector>

#include "include/constants.h"
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

class TAddCSI;

template <>
void WriteStream<TAddCSI>(std::ostream& stream, const TAddCSI& node);

class TAddCSI final : public TOp {
 public:
  TAddCSI(const TensorLayout& layout, const ScaledPtVal& scalar);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_dag,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }

  virtual std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TAddCSI>(*this);
  }
  const ScaledPtVal& Scalar() const { return scalar_; }

  LogScale AddedLogScale() const final { LOG(FATAL); }
  int BackendMaskDepth() const final { LOG(FATAL); }

  const std::string& TypeName() const { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TAddCSI>(stream, *this);
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

  static TOpDerivedRegistrar<TAddCSI> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TAddCSI::TAddCSI(const TensorLayout& layout, const ScaledPtVal& scalar)
    : layout_(layout), scalar_(scalar) {}

inline TOpDerivedRegistrar<TAddCSI> TAddCSI::reg_{TAddCSI::StaticTypeName()};

template <>
inline void WriteStream<TAddCSI>(std::ostream& stream, const TAddCSI& node) {
  WriteStream<std::string>(stream, TAddCSI::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<ScaledPtVal>(stream, node.Scalar());
}

template <>
inline TAddCSI ReadStreamWithoutTypeNamePrefix<TAddCSI>(std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto scaled_pt_val = ReadStream<ScaledPtVal>(stream);
  return {tensor_layout, scaled_pt_val};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_ADD_CSI_H_
