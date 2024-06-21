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

#ifndef FHELIPE_CT_OP_H_
#define FHELIPE_CT_OP_H_

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "level.h"
#include "level_info.h"
#include "log_scale.h"

namespace fhelipe {

class CtOpVisitor;

class CtOp {
 public:
  typedef std::unordered_map<std::string,
                             std::unique_ptr<CtOp> (*)(std::istream&)>
      DerivedRecordType;
  explicit CtOp(const LevelInfo& level_info) : level_info_(level_info) {}
  CtOp(CtOp&&) = default;
  virtual ~CtOp() = default;
  CtOp(const CtOp&) = delete;
  CtOp& operator=(const CtOp&) = delete;
  CtOp& operator=(CtOp&&) = default;

  virtual const std::string& TypeName() const = 0;

  Level GetLevel() const { return level_info_.Level(); }
  LogScale LogScale() const { return level_info_.LogScale(); }
  const LevelInfo& GetLevelInfo() const { return level_info_; }
  void SetLevelInfo(const LevelInfo& level_info) { level_info_ = level_info; }

  virtual void WriteStreamHelper(std::ostream& stream) const = 0;
  virtual std::unique_ptr<CtOp> CloneUniq() const = 0;
  static std::unique_ptr<CtOp> CreateInstance(std::istream& stream);

 protected:
  static DerivedRecordType& GetMap() {
    static DerivedRecordType record_map_;
    return record_map_;
  }

 private:
  LevelInfo level_info_;
};

// https://stackoverflow.com/questions/582331/is-there-a-way-to-instantiate-objects-from-a-string-holding-their-class-name
template <class T>
class CtOpDerivedRegistrar : public CtOp {
 public:
  explicit CtOpDerivedRegistrar(const std::string& type_name);
  void WriteStreamHelper(std::ostream& stream) const final {
    (void)stream;
    LOG(FATAL);
  }
  std::unique_ptr<CtOp> CloneUniq() const final { LOG(FATAL); }
  const std::string& TypeName() const final { LOG(FATAL); }

 private:
  std::string type_name_;
};

template <>
inline void WriteStream<CtOp>(std::ostream& stream, const CtOp& ct_op) {
  ct_op.WriteStreamHelper(stream);
}

template <class T>
std::enable_if_t<std::is_base_of_v<CtOp, T>, T> ReadStreamWithoutTypeNamePrefix(
    std::istream& stream);

template <class T>
inline std::enable_if_t<std::is_base_of_v<CtOp, T>, T> ReadStream(
    std::istream& stream) {
  auto token = ReadStream<std::string>(stream);
  CHECK(token == T::TypeName());
  return ReadStreamWithoutTypeNamePrefix<T>(stream);
}

template <class T>
CtOpDerivedRegistrar<T>::CtOpDerivedRegistrar(const std::string& type_name)
    : CtOp(LevelInfo(5, 50)), type_name_(type_name) {
  GetMap().emplace(std::string(type_name), [](std::istream& stream) {
    return std::unique_ptr<CtOp>{
        new T{ReadStreamWithoutTypeNamePrefix<T>(stream)}};
  });
}

}  // namespace fhelipe

#endif  // FHELIPE_CT_OP_H_
