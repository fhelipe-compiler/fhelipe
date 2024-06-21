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

#ifndef FHELIPE_T_BOOTSTRAP_C_H_
#define FHELIPE_T_BOOTSTRAP_C_H_

#include <memory>
#include <ostream>
#include <vector>

#include "ct_op.h"
#include "include/tensor_layout.h"
#include "laid_out_tensor.h"
#include "t_op.h"

namespace fhelipe {
class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TBootstrapC;

template <>
void WriteStream<TBootstrapC>(std::ostream& stream, const TBootstrapC& node);

class TBootstrapC final : public TOp {
 public:
  TBootstrapC(const TensorLayout& layout, const Level& usable_levels,
              std::optional<bool> is_shortcut = std::nullopt);
  LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_dag,
      const std::vector<LaidOutTensorCt>& input_tensor) const final;
  std::unique_ptr<TOp> CloneUniq() const final {
    return std::make_unique<TBootstrapC>(*this);
  }

  const TensorLayout& OutputLayout() const final { return layout_; }

  const Level& GetUsableLevels() const { return usable_levels_; }

  LogScale AddedLogScale() const final { return 0; }
  int BackendMaskDepth() const final { return 0; }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) final;

  const std::string& TypeName() const { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TBootstrapC>(stream, *this);
  }

  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "TBootstrapC";
    return type_name_;
  }

  std::optional<bool> IsShortcut() const { return is_shortcut_; }

 private:
  TensorLayout layout_;
  Level usable_levels_;
  std::optional<bool> is_shortcut_;

  static TOpDerivedRegistrar<TBootstrapC> reg_;
  bool EqualTo(const TOp& other) const final;
};

inline TOpDerivedRegistrar<TBootstrapC> TBootstrapC::reg_{
    TBootstrapC::StaticTypeName()};

template <>
inline void WriteStream<TBootstrapC>(std::ostream& stream,
                                     const TBootstrapC& node) {
  WriteStream<std::string>(stream, TBootstrapC::StaticTypeName());
  stream << " ";
  WriteStream<TensorLayout>(stream, node.OutputLayout());
  stream << " ";
  WriteStream<Level>(stream, node.GetUsableLevels());
  stream << " ";
  WriteStream(stream, node.IsShortcut());
}

template <>
inline TBootstrapC ReadStreamWithoutTypeNamePrefix<TBootstrapC>(
    std::istream& stream) {
  auto tensor_layout = ReadStream<TensorLayout>(stream);
  auto usable_levels = ReadStream<Level>(stream);
  auto is_shortcut = ReadStream<std::optional<bool>>(stream);
  return {tensor_layout, usable_levels, is_shortcut};
}

}  // namespace fhelipe

#endif  // FHELIPE_T_BOOTSTRAP_C_H_
