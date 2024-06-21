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

#ifndef FHELIPE_T_MUL_CP_H_
#define FHELIPE_T_MUL_CP_H_

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

class TMulCP;

template <>
void WriteStream<TMulCP>(std::ostream& stream, const TMulCP& node);

class TMulCP final : public TOp {
 public:
  TMulCP(const TensorLayout& layout, const std::string& pt_tensor_name,
         LogScale pt_tensor_log_scale);
  TOp::LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<TOp::LaidOutTensorCt>& input_tensors) const final;
  const TensorLayout& OutputLayout() const final { return layout_; }
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TMulCP>(*this);
  }
  const std::string& PtTensorName() const { return pt_tensor_name_; }
  LogScale PtTensorLogScale() const { return pt_tensor_log_scale_; }

  LogScale AddedLogScale() const final { return pt_tensor_log_scale_; }
  int BackendMaskDepth() const final { return 0; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TMulCP>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TMulCP";
    return type_name_;
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

 private:
  TensorLayout layout_;
  std::string pt_tensor_name_;
  LogScale pt_tensor_log_scale_;

  static TOpDerivedRegistrar<TMulCP> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TMulCP::TMulCP(const TensorLayout& layout,
                      const std::string& pt_tensor_name,
                      LogScale pt_tensor_log_scale)
    : layout_(layout),
      pt_tensor_name_(pt_tensor_name),
      pt_tensor_log_scale_(pt_tensor_log_scale) {}

inline TOpDerivedRegistrar<TMulCP> TMulCP::reg_{TMulCP::StaticTypeName()};

template <>
inline void WriteStream<TMulCP>(std::ostream& stream, const TMulCP& node) {
  WriteStream<std::string>(stream, TMulCP::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<std::string>(stream, node.PtTensorName());
  stream << " ";
  WriteStream<LogScale>(stream, node.PtTensorLogScale());
}

template <>
inline TMulCP ReadStreamWithoutTypeNamePrefix<TMulCP>(std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto pt_tensor_name = ReadStream<std::string>(stream);
  auto pt_tensor_log_scale = ReadStream<LogScale>(stream);
  return {tensor_layout, pt_tensor_name, pt_tensor_log_scale};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_MUL_CP_H_
