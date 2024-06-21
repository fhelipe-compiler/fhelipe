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

#ifndef FHELIPE_ADD_CP_H_
#define FHELIPE_ADD_CP_H_

#include <memory>
#include <sstream>
#include <vector>

#include "ct_op.h"
#include "ct_op_visitor.h"
#include "dictionary.h"
#include "log_scale.h"

namespace fhelipe {

class CtOpVisitor;

template <>
void WriteStream<AddCP>(std::ostream& stream, const AddCP& node);

class AddCP final : public CtOp {
 public:
  AddCP(const LevelInfo& level_info, const KeyType& handle,
        const class LogScale& pt_log_scale);

  const KeyType& GetHandle() const;
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<AddCP>(GetLevelInfo(), handle_, pt_log_scale_);
  }
  class LogScale GetPtLogScale() const { return pt_log_scale_; }
  const std::string& TypeName() const final { return StaticTypeName(); }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<AddCP>(stream, *this);
  }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "AddCP";
    return type_name_;
  }

 private:
  KeyType handle_;
  class LogScale pt_log_scale_;

  static const std::string type_name_;
  static CtOpDerivedRegistrar<AddCP> reg_;
};

inline AddCP::AddCP(const LevelInfo& level_info, const KeyType& handle,
                    const class LogScale& pt_log_scale)
    : CtOp(level_info), handle_(handle), pt_log_scale_(pt_log_scale) {}

inline const KeyType& AddCP::GetHandle() const { return handle_; }

inline CtOpDerivedRegistrar<AddCP> AddCP::reg_{AddCP::StaticTypeName()};

template <>
inline void WriteStream<AddCP>(std::ostream& stream, const AddCP& node) {
  WriteStream<std::string>(stream, AddCP::StaticTypeName());
  stream << " ";
  WriteStream<KeyType>(stream, node.GetHandle());
  stream << " ";
  WriteStream<class LogScale>(stream, node.GetPtLogScale());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline AddCP ReadStreamWithoutTypeNamePrefix<AddCP>(std::istream& stream) {
  auto handle = ReadStream<KeyType>(stream);
  auto pt_log_scale = ReadStream<class LogScale>(stream);
  auto level_info = ReadStream<LevelInfo>(stream);
  return {level_info, handle, pt_log_scale};
}

}  // namespace fhelipe

#endif  // FHELIPE_ADD_CP_H_
