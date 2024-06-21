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

#ifndef FHELIPE_T_RESCALE_C_H_
#define FHELIPE_T_RESCALE_C_H_

#include <memory>

#include "include/constants.h"
#include "log_scale.h"
#include "t_op.h"

namespace fhelipe {

class TRescaleC;

template <>
void WriteStream<TRescaleC>(std::ostream& stream, const TRescaleC& node);

class TRescaleC final : public TOp {
 public:
  TRescaleC(const TensorLayout& layout, LogScale rescale_amount)
      : layout_(layout), rescale_amount_(rescale_amount) {}
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const final;

  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TRescaleC>(*this);
  }
  LogScale AddedLogScale() const final { LOG(FATAL); }
  int BackendMaskDepth() const final { LOG(FATAL); }

  const TensorLayout& OutputLayout() const final { return layout_; }

  LogScale RescaleAmount() const { return rescale_amount_; }

  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TRescaleC>(stream, *this);
  }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TRescaleC";
    return type_name_;
  }

 private:
  TensorLayout layout_;
  LogScale rescale_amount_;

  static TOpDerivedRegistrar<TRescaleC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TRescaleC> TRescaleC::reg_{
    TRescaleC::StaticTypeName()};

template <>
inline void WriteStream<TRescaleC>(std::ostream& stream,
                                   const TRescaleC& node) {
  WriteStream<std::string>(stream, TRescaleC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<LogScale>(stream, node.RescaleAmount());
}

template <>
inline TRescaleC ReadStreamWithoutTypeNamePrefix<TRescaleC>(
    std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto rescale_amount = ReadStream<LogScale>(stream);
  return {tensor_layout, rescale_amount};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_RESCALE_C_H_
