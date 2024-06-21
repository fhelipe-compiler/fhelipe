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

#ifndef FHELIPE_ADD_CS_H_
#define FHELIPE_ADD_CS_H_

#include <iomanip>
#include <string>
#include <vector>

#include "constants.h"
#include "ct_op.h"
#include "ct_op_visitor.h"
#include "scaled_pt_val.h"

namespace fhelipe {

template <>
void WriteStream<AddCS>(std::ostream& stream, const AddCS& node);

class AddCS final : public CtOp {
 public:
  AddCS(const LevelInfo& level_info, const ScaledPtVal& scalar);

  ScaledPtVal Scalar() const;
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<AddCS>(GetLevelInfo(), scalar_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<AddCS>(stream, *this);
  }
  const std::string& TypeName() const final { return StaticTypeName(); }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "AddCS";
    return type_name_;
  }

 private:
  ScaledPtVal scalar_;

  static CtOpDerivedRegistrar<AddCS> reg_;
};

inline AddCS::AddCS(const LevelInfo& level_info, const ScaledPtVal& scalar)
    : CtOp(level_info), scalar_(scalar) {}

inline ScaledPtVal AddCS::Scalar() const { return scalar_; }

inline CtOpDerivedRegistrar<AddCS> AddCS::reg_{AddCS::StaticTypeName()};

template <>
inline void WriteStream<AddCS>(std::ostream& stream, const AddCS& node) {
  WriteStream<std::string>(stream, AddCS::StaticTypeName());
  stream << " ";
  WriteStream<ScaledPtVal>(stream, node.Scalar());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline AddCS ReadStreamWithoutTypeNamePrefix<AddCS>(std::istream& stream) {
  auto scalar = ReadStream<ScaledPtVal>(stream);
  auto level_info = ReadStream<LevelInfo>(stream);
  return {level_info, scalar};
}

}  // namespace fhelipe

#endif  // FHELIPE_ADD_CS_H_
