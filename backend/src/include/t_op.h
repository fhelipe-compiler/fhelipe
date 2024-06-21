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

#ifndef FHELIPE_T_OP_H_
#define FHELIPE_T_OP_H_

#include <memory>
#include <ostream>
#include <vector>

#include "laid_out_tensor.h"
#include "node.h"
#include "tensor_layout.h"

namespace fhelipe {

class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

class TOp {
 public:
  typedef std::unordered_map<std::string,
                             std::unique_ptr<TOp> (*)(std::istream&)>
      DerivedRecordType;
  typedef std::shared_ptr<Node<CtOp>> Chunk;
  typedef LaidOutChunk<Chunk> LaidOutChunk;
  typedef LaidOutTensor<Chunk> LaidOutTensorCt;

  TOp() = default;
  TOp(TOp&&) = default;
  virtual ~TOp() {}
  TOp(const TOp& t_op) = default;
  TOp& operator=(const TOp&) = delete;
  TOp& operator=(TOp&&) = default;

  virtual std::unique_ptr<TOp> CloneUniq() const = 0;
  virtual const TensorLayout& OutputLayout() const = 0;
  // TODO(nsamar): Remove ct_program as an argument here
  virtual LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<LaidOutTensorCt>& input_tensors) const = 0;
  virtual void WriteStreamHelper(std::ostream& stream) const = 0;

  virtual LogScale AddedLogScale() const { return 0; }
  virtual int BackendMaskDepth() const { return 0; }

  virtual const std::string& TypeName() const = 0;
  static std::unique_ptr<TOp> CreateInstance(std::istream& stream);
  virtual void SetLayouts(const TensorLayout& input_layout,
                          const TensorLayout& output_layout) = 0;
  friend bool operator==(const TOp& lhs, const TOp& rhs) {
    return lhs.EqualTo(rhs);
  }

 protected:
  static DerivedRecordType& GetMap() {
    static DerivedRecordType record_map_ = TOp::DerivedRecordType{};
    return record_map_;
  }

 private:
  virtual bool EqualTo(const TOp& other) const = 0;
};

// https://stackoverflow.com/questions/582331/is-there-a-way-to-instantiate-objects-from-a-string-holding-their-class-name
template <class T>
class TOpDerivedRegistrar : public TOp {
 public:
  explicit TOpDerivedRegistrar(const std::string& type_name);
  std::unique_ptr<TOp> CloneUniq() const final { LOG(FATAL); }
  const TensorLayout& OutputLayout() const final { LOG(FATAL); }
  void SetLayouts(const TensorLayout& input_layout,
                  const TensorLayout& output_layout) {
    LOG(FATAL);
  }
  TOp::LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_dag,
      const std::vector<TOp::LaidOutTensorCt>& input_tensors) const final {
    (void)ct_dag;
    (void)input_tensors;
    LOG(FATAL);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    (void)stream;
    LOG(FATAL);
  }
  const std::string& TypeName() const final { LOG(FATAL); }

 private:
  std::string type_name_;

  bool EqualTo(const TOp& other) const { LOG(FATAL); }
};

template <>
inline void WriteStream<TOp>(std::ostream& stream, const TOp& t_op) {
  t_op.WriteStreamHelper(stream);
}

template <class T>
std::enable_if_t<std::is_base_of_v<TOp, T>, T> ReadStreamWithoutTypeNamePrefix(
    std::istream& stream);

template <class T>
inline std::enable_if_t<std::is_base_of_v<TOp, T>, T> ReadStream(
    std::istream& stream) {
  auto token = ReadStream<std::string>(stream);
  CHECK(token == T::StaticTypeName());
  return ReadStreamWithoutTypeNamePrefix<T>(stream);
}

template <class T>
TOpDerivedRegistrar<T>::TOpDerivedRegistrar(const std::string& type_name)
    : type_name_(type_name) {
  GetMap().emplace(std::string(type_name), [](std::istream& stream) {
    return std::unique_ptr<TOp>{
        new T{std::move(ReadStreamWithoutTypeNamePrefix<T>(stream))}};
  });
}

namespace detail {

template <>
inline void CheckLevelInfo<TOp::Chunk>(
    const std::vector<LaidOutChunk<TOp::Chunk>>& locs) {
  for (const auto& loc : locs) {
    CHECK(loc.Chunk()->Value().GetLevelInfo() ==
          locs.at(0).Chunk()->Value().GetLevelInfo());
  }
}

}  // namespace detail

}  // namespace fhelipe

#endif  // FHELIPE_T_OP_H_
