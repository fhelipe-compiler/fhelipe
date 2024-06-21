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

#ifndef FHELIPE_MUL_CS_H_
#define FHELIPE_MUL_CS_H_

#include <memory>
#include <string>
#include <vector>

#include "ct_op.h"
#include "ct_op_visitor.h"
#include "level_info.h"
#include "plaintext.h"
#include "scaled_pt_val.h"

namespace fhelipe {

class CtOpVisitor;

template <>
void WriteStream<MulCS>(std::ostream& stream, const MulCS& node);

class MulCS final : public CtOp {
 public:
  MulCS(const LevelInfo& level_info, const ScaledPtVal& scalar);

  const ScaledPtVal& Scalar() const;
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<MulCS>(GetLevelInfo(), scalar_);
  }
  const std::string& TypeName() const final { return StaticTypeName(); }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "MulCS";
    return type_name_;
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<MulCS>(stream, *this);
  }

 private:
  ScaledPtVal scalar_;

  static const std::string type_name_;
  static CtOpDerivedRegistrar<MulCS> reg_;
};

inline MulCS::MulCS(const LevelInfo& level_info, const ScaledPtVal& scalar)
    : CtOp(level_info), scalar_(scalar) {}

inline const ScaledPtVal& MulCS::Scalar() const { return scalar_; }

inline CtOpDerivedRegistrar<MulCS> MulCS::reg_{MulCS::StaticTypeName()};

template <>
inline void WriteStream<MulCS>(std::ostream& stream, const MulCS& node) {
  WriteStream<std::string>(stream, MulCS::StaticTypeName());
  stream << " ";
  WriteStream<ScaledPtVal>(stream, node.Scalar());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline MulCS ReadStreamWithoutTypeNamePrefix<MulCS>(std::istream& stream) {
  auto scalar = ReadStream<ScaledPtVal>(stream);
  auto level_info = ReadStream<LevelInfo>(stream);
  return {level_info, scalar};
}

}  // namespace fhelipe

#endif  // FHELIPE_MUL_CS_H_
