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

#ifndef FHELIPE_LEVELED_T_OP_H_
#define FHELIPE_LEVELED_T_OP_H_

#include <memory>

#include "level_info.h"
#include "t_input_c.h"
#include "t_op.h"

namespace fhelipe {

namespace detail {

inline void CheckLevelInfo(const std::vector<LaidOutChunk<TOp::Chunk>>& locs,
                           const LevelInfo& expected_level_info) {
  for (const auto& loc : locs) {
    CHECK(loc.Chunk()->Value().GetLevelInfo() == expected_level_info);
  }
}

}  // namespace detail

class LeveledTOp {
 public:
  LeveledTOp(std::unique_ptr<TOp>&& t_op, const LevelInfo& level_info,
             std::optional<int> depth = std::nullopt)
      : t_op_(std::move(t_op)), level_info_(level_info), depth_(depth) {}

  TOp::LaidOutTensorCt AmendCtProgram(
      ct_program::CtProgram& ct_program,
      const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
    if (const TInputC* t_input_c = dynamic_cast<const TInputC*>(t_op_.get())) {
      const TOp::LaidOutTensorCt& result =
          t_input_c->CreateInputTensor(ct_program, level_info_);
      detail::CheckLevelInfo(result.Chunks(), level_info_);
      return result;
    }

    const TOp::LaidOutTensorCt& result =
        t_op_->AmendCtProgram(ct_program, input_tensors);
    detail::CheckLevelInfo(result.Chunks(), level_info_);
    return result;
  }

  void SetLevelInfo(const LevelInfo& level_info) { level_info_ = level_info; }

  const LevelInfo& GetLevelInfo() const { return level_info_; }
  class Level Level() const { return level_info_.Level(); }
  std::optional<int> Depth() const { return depth_; }
  class LogScale LogScale() const { return level_info_.LogScale(); }
  const TOp& GetTOp() const { return *t_op_; }
  std::unique_ptr<LeveledTOp> CloneUniq() const {
    return std::make_unique<LeveledTOp>(t_op_->CloneUniq(), level_info_,
                                        depth_);
  }

 private:
  std::unique_ptr<TOp> t_op_;
  LevelInfo level_info_;
  std::optional<int> depth_;
};

template <>
inline void WriteStream<LeveledTOp>(std::ostream& stream,
                                    const LeveledTOp& t_op) {
  WriteStream(stream, t_op.GetTOp());
  stream << " ";
  WriteStream(stream, t_op.GetLevelInfo());
  WriteStream(stream, t_op.Depth());
}

template <>
inline LeveledTOp ReadStream<LeveledTOp>(std::istream& stream) {
  auto t_op = TOp::CreateInstance(stream);
  auto level_info = ReadStream<LevelInfo>(stream);
  auto depth = ReadStream<std::optional<int>>(stream);
  return {std::move(t_op), level_info, depth};
}

inline std::ostream& operator<<(std::ostream& stream, const LeveledTOp& t_op) {
  WriteStream<LeveledTOp>(stream, t_op);
  return stream;
}

}  // namespace fhelipe

#endif  // FHELIPE_LEVELED_T_OP_H_
