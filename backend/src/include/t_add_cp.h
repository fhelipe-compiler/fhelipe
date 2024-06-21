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

#ifndef FHELIPE_T_ADD_CP_H_
#define FHELIPE_T_ADD_CP_H_

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "include/constants.h"
#include "laid_out_tensor.h"
#include "shape.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {
class CtOp;

namespace ct_dag {
class CtProgram;
}  // namespace ct_dag

class TAddCP;

template <>
void WriteStream<TAddCP>(std::ostream& stream, const TAddCP& node);

class TAddCP final : public TOp {
 public:
  explicit TAddCP(const TensorLayout& layout, const std::string& pt_tensor_name,
                  LogScale pt_tensor_log_scale);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_dag,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TAddCP>(*this);
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  const std::string& PtTensorName() const { return pt_tensor_name_; }
  LogScale PtTensorLogScale() const { return pt_tensor_log_scale_; }

  LogScale AddedLogScale() const final { LOG(FATAL); }
  int BackendMaskDepth() const final { LOG(FATAL); }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TAddCP>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TAddCP";
    return type_name_;
  }

 private:
  TensorLayout layout_;
  std::string pt_tensor_name_;
  LogScale pt_tensor_log_scale_;

  static TOpDerivedRegistrar<TAddCP> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TAddCP::TAddCP(const TensorLayout& layout,
                      const std::string& pt_tensor_name,
                      LogScale pt_tensor_log_scale)
    : layout_(layout),
      pt_tensor_name_(pt_tensor_name),
      pt_tensor_log_scale_(pt_tensor_log_scale) {}

inline TOpDerivedRegistrar<TAddCP> TAddCP::reg_{TAddCP::StaticTypeName()};

template <>
inline void WriteStream<TAddCP>(std::ostream& stream, const TAddCP& node) {
  WriteStream<std::string>(stream, TAddCP::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<std::string>(stream, node.PtTensorName());
  stream << " ";
  WriteStream<LogScale>(stream, node.PtTensorLogScale());
}

template <>
inline TAddCP ReadStreamWithoutTypeNamePrefix<TAddCP>(std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto pt_tensor_name = ReadStream<std::string>(stream);
  auto pt_tensor_log_scale = ReadStream<LogScale>(stream);
  return TAddCP(tensor_layout, pt_tensor_name, pt_tensor_log_scale);
}

}  // namespace fhelipe

#endif  // FHELIPE_T_ADD_CP_H_
