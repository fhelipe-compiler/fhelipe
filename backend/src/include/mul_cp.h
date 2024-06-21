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

#ifndef FHELIPE_MUL_CP_H_
#define FHELIPE_MUL_CP_H_

#include <memory>
#include <vector>

#include "ct_op.h"
#include "dictionary.h"
#include "io_utils.h"
#include "level_info.h"
#include "log_scale.h"

namespace fhelipe {

class MulCP;

template <>
void WriteStream<MulCP>(std::ostream& stream, const MulCP& node);

class MulCP : public CtOp {
 public:
  MulCP(const LevelInfo& level_info, const KeyType& handle,
        const class LogScale& pt_log_scale);

  const KeyType& GetHandle() const;
  std::unique_ptr<CtOp> CloneUniq() const final {
    return std::make_unique<MulCP>(GetLevelInfo(), handle_, pt_log_scale_);
  }
  class LogScale GetPtLogScale() const { return pt_log_scale_; }
  const std::string& TypeName() const final { return StaticTypeName(); }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<MulCP>(stream, *this);
  }
  static const std::string& StaticTypeName() {
    static const std::string type_name_ = "MulCP";
    return type_name_;
  }

 private:
  KeyType handle_;
  class LogScale pt_log_scale_;
  void WriteToStream(std::ostream& stream) const;

  static const std::string type_name_;
  static CtOpDerivedRegistrar<MulCP> reg_;
};

inline MulCP::MulCP(const LevelInfo& level_info, const KeyType& handle,
                    const class LogScale& pt_log_scale)
    : CtOp(level_info), handle_(handle), pt_log_scale_(pt_log_scale) {}

inline const KeyType& MulCP::GetHandle() const { return handle_; }

inline CtOpDerivedRegistrar<MulCP> MulCP::reg_{MulCP::StaticTypeName()};

template <>
inline void WriteStream<MulCP>(std::ostream& stream, const MulCP& node) {
  WriteStream(stream, MulCP::StaticTypeName());
  stream << " ";
  WriteStream<KeyType>(stream, node.GetHandle());
  stream << " ";
  WriteStream<class LogScale>(stream, node.GetPtLogScale());
  stream << " ";
  WriteStream<LevelInfo>(stream, node.GetLevelInfo());
}

template <>
inline MulCP ReadStreamWithoutTypeNamePrefix<MulCP>(std::istream& stream) {
  auto handle = ReadStream<KeyType>(stream);
  auto pt_log_scale = ReadStream<class LogScale>(stream);
  auto level_info = ReadStream<LevelInfo>(stream);
  return {level_info, handle, pt_log_scale};
}

}  // namespace fhelipe

#endif  // FHELIPE_MUL_CP_H_
