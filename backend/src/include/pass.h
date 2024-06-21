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

#ifndef FHELIPE_PASS_H_
#define FHELIPE_PASS_H_

#include "dag.h"

namespace fhelipe {

class PassName {
 public:
  explicit PassName(const std::string& pass_name) : pass_name_(pass_name) {}
  friend bool operator==(const PassName& lhs, const PassName& rhs);
  const std::string& String() const { return pass_name_; }

 private:
  std::string pass_name_;
};

template <>
inline void WriteStream<PassName>(std::ostream& stream,
                                  const PassName& pass_name) {
  WriteStream(stream, pass_name.String());
}

template <>
inline PassName ReadStream(std::istream& stream) {
  return PassName{ReadStream<std::string>(stream)};
}

inline bool operator==(const PassName& lhs, const PassName& rhs) {
  return lhs.pass_name_ == rhs.pass_name_;
}

template <class InputT, class OutputT>
class Pass {
 public:
  Pass() = default;
  Pass(const Pass&) = default;
  Pass(Pass&&) noexcept = default;

  virtual OutputT DoPass(const InputT& old_dag) = 0;
  virtual const PassName& GetPassName() const = 0;
  virtual std::unique_ptr<Pass<InputT, OutputT>> CloneUniq() const = 0;
  virtual ~Pass() = default;
};

}  // namespace fhelipe

#endif  // FHELIPE_PASS_H_
